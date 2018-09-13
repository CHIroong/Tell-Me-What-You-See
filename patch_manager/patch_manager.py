import json
import os
import shutil

import numpy as np
from skimage import io

from PIL import Image
from konlpy.tag import Okt

from patch_manager.lexrankr import LexRank
from patch_manager.dom_feature_extracter import DOMFeatureExtracter
from patch_manager.easylist import EasyListHandler
from classifier.classifier import PatchClassifier

class PatchManager:
    def __init__(self, patch_size=96):
        self.data = []
        self.patch_size = patch_size
        self.patch_directory = None
        self.spec_filename = None
        self.tagged = None
        self.easylist = EasyListHandler()
        self.okt = Okt()
        self.lexrank = LexRank(similarity='jaccard')
        self.classifier = PatchClassifier(use_gpu=False)

    def feed(self, data_id, filename_image, filename_html):
        self.data.append({
            "id": str(data_id),
            "filename_img": filename_image,
            "filename_html": filename_html,
            "dom": DOMFeatureExtracter(filename_html, self.easylist, self.okt, self.lexrank)
        })
    
    def is_valid(self, filename_image, filename_html, verbose=False):
        dom = DOMFeatureExtracter(filename_html)
        img = Image.open(filename_image)
        dom_width, dom_height = dom.size()
        img_width, img_height = img.size
        if abs(dom_width-img_width) > 50 or abs(dom_height-img_height) > 500:
            if verbose:
                print("Image %s is %dx%d but DOM is %dx%d" % (filename_image, img_width, img_height, dom_width, dom_height))
            return False
        return True

    def feed_tagged_data(self, tagged_data_filename):
        with open(tagged_data_filename, 'r') as f:
            self.tagged = json.loads(f.read())
    
    def tagged_data_of_id(self, data_id):
        for screenshot_id, rects in self.tagged:
            if str(screenshot_id) == str(data_id):
                return rects
        return []

    def save_patches_at(self, patch_directory):
        if patch_directory[-1] != '/':
            patch_directory += '/'
        self.patch_directory = patch_directory
    
    def save_spec_at(self, spec_filename):
        self.spec_filename = spec_filename

    def generate_spec(self, verbose=False):
        if self.patch_directory is None:
            raise Exception("Patch directory is not specified")

        if self.patch_directory[:-1] not in os.listdir():
            os.mkdir(self.patch_directory)
        for data in self.data:
            if data["id"] not in os.listdir(self.patch_directory):
                os.mkdir(self.patch_directory + "%s/" % data["id"])

        result_spec = {
            "width": self.patch_size,
            "height": self.patch_size,
            "root": self.patch_directory,
            "tags": [ {"name": "text"}, {"name": "image"}, {"name": "graph"}, {"name": "ad"}, {"name": "xad"}, {"name": "nothing"}, ],
            "data": []
        }

        for data in self.data:
            if verbose:
                print(f'Generating patches of {data["filename_img"]}')
            
            dom = data["dom"]
            cell = {
                "id": data["id"],
                "filename": data["filename_img"].split('/')[-1],
                "width": 0, "height": 0,
                "doc_features": {
                    "keywords": dom.salient_keywords(),
                    "title": dom.title(),
                    "num_imgs": dom.num_tags("img"),
                    "num_headers": dom.num_tags("h1,h2,h3,h4,h5,h6"),
                    "num_links": dom.num_tags("a"),
                    "num_tables": dom.num_tags("table"),
                    "navs": dom.navs(),
                    "headers": dom.headers(),
                    "salient_sentences": dom.salient_sentences(),
                    "summary": dom.summary(),
                },
                "patches": []
            }

            print("====salient sentences====")
            print('\n----\n'.join(cell["doc_features"]["salient_sentences"]))
            print("====summary====")
            print('\n----\n'.join(cell["doc_features"]["summary"]))
            print("========")

            folder = self.patch_directory + "%s/" % data["id"]
            shutil.copyfile(data["filename_img"], folder + cell["filename"])
            img = Image.open(folder + cell["filename"])
            dom_width, dom_height = data["dom"].size()
            cell["width"], cell["height"] = img.size
            if dom_height < cell["height"]:
                cell["height"] = dom_height

            if self.tagged is not None:
                rects = self.tagged_data_of_id(data["id"])

            if verbose:
                prev = 20

            for i in range(0, cell["width"]-self.patch_size, self.patch_size):
                for j in range(0, cell["height"]-self.patch_size, self.patch_size):
                    left, top, right, bottom = i, j, i + self.patch_size - 1, j + self.patch_size - 1
                    cropped_filename = "%dx%d.png" % (i//self.patch_size, j//self.patch_size)
                    cropped = img.crop((left, top, right + 1, bottom + 1))
                    cropped.save(folder + cropped_filename)

                    dom = data["dom"]
                    x, y = i + self.patch_size // 2, j + self.patch_size // 2
                    features = {
                        "cursor": dom.cursor_style(x, y),
                        "aspect_ratio": dom.element_aspect_ratio(x, y),
                        "is_img": dom.is_img(x, y),
                        "is_iframe": dom.is_iframe(x, y),
                        "nested_a_tags": dom.num_nested_a_tags(x, y),
                        "contains_harmful_url": dom.has_harmful_url_segment(x, y),
                    }

                    patch = {
                        "filename": cropped_filename,
                        "left": i,
                        "top": j,
                        "right": i + self.patch_size - 1,
                        "bottom": j + self.patch_size - 1,
                        "tags" : [0] * len(result_spec["tags"]),
                        "features": features,
                    }


                    if self.tagged is None:
                        self.classify_tags(patch["tags"], folder + cropped_filename, patch, dom_width, dom_height)
                    else:
                        self.calc_tags(rects, patch["tags"], (left, top, right, bottom))

                    cell["patches"].append(patch)

                    if verbose and (i*cell["height"]+j)/cell["height"]/cell["width"] > prev/100:
                        print(f"... patch making {prev}% done")
                        prev += 20
            result_spec["data"].append(cell)
        
        with open(self.spec_filename, "w") as f:
            f.write(json.dumps(result_spec, indent=4))

    def calc_tags(self, rects, tags, patch):
        l, u, r, d = patch
        patch_area = (r - l) * (d - u)
        for rect in rects:
            try:
                ll = rect["left"]
                uu = rect["top"]
                rr = rect["left"] + rect["width"]
                dd = rect["top"] + rect["height"]
            except TypeError as e: # sometimes None values are in the rect
                continue
            area = max(0, min(r, rr) - max(l, ll)) * max(0, min(d, dd) - max(u, uu))
            tags[rect["type_id"] - 1] += area / patch_area
        tags[-1] = 1 - sum(tags)
    
    def classify_tags(self, tags, patch_filename, patch, width, height):
        patch_image = np.moveaxis(io.imread(patch_filename), -1, 0)
        patch_image[3, 0, 0] = patch["left"] / width * 100
        patch_image[3, 0, 1] = patch["top"] / height * 100
        patch_image[3, 0, 2] = abs(50-patch_image[3, 0, 0]) * 2 # normalized min distance from either left or right edges (0 - 100), e.g., abs(50 - norm_left) * 2
        patch_image[3, 0, 3] = abs(50-patch_image[3, 0, 1]) * 2 # normalized min distance from either top or bottom edges (0 - 100), e.g., abs(50 - norm_top) * 2
        patch_image[3, 0, 4] = float(patch["features"]["cursor"] == "pointer")
        patch_image[3, 0, 5] = min(255, patch["features"]["aspect_ratio"])
        patch_image[3, 0, 6] = float(patch["features"]["is_img"])
        patch_image[3, 0, 7] = float(patch["features"]["is_iframe"])
        patch_image[3, 0, 8] = patch["features"]["nested_a_tags"]
        patch_image[3, 0, 9] = float(patch["features"]["contains_harmful_url"])
        scores = self.classifier.classify_one(patch_image)
        tags[np.argmax(scores)] += 1

        """
        vis = io.imread(folder + data["filename"])
        if self.verbose:
            colors = np.array([
                    [173, 181, 189, 255],
                    [64, 192, 87, 255],
                    [92, 124, 250, 255],
                    [252, 196, 25, 255]
                ])
            if np.argmax(scores) < 4:
                confidence = np.max(scores)
                j, i = map(int, patch["filename"].split('.')[0].split('x'))
                i *= self.patch_size
                j *= self.patch_size
                vis[i:i+self.patch_size,j:j+self.patch_size,:] =  \
                    ((1-confidence) * vis[i:i+self.patch_size,j:j+self.patch_size,:] + \
                    confidence * colors[np.argmax(scores)]).astype('uint8')

        if self.verbose:
            print(self.tags)
            io.imsave('_'.join((folder + data["filename"]).split('.')[:-1]) + '_vis.png', vis)

        """