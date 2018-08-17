import os
import sys
import json
import shutil
import pprint
from collections import defaultdict

import requests
from PIL import Image

url = sys.argv[1]
data = json.loads(requests.get(url).text)

PATCH_SIZE = 128

rects_info = defaultdict(list)
for rect in data["target"]:
    rects_info[rect["screenshot_id"]].append(rect)

result_spec = {
    "width": PATCH_SIZE,
    "height": PATCH_SIZE,
    "root": 'images/',
    "tags": [
        {"name": "text"},
        {"name": "image"},
        {"name": "graph"},
        {"name": "ad"},
        {"name": "xad"},
        {"name": "nothing"},
    ],
    "data": []
}

if result_spec["root"][:-1] not in os.listdir():
    os.mkdir(result_spec["root"])

def calc_tags(tags, rects, patch):
    l, u, r, d = patch
    for rect in rects:
        try:
            ll = rect["left"]
            uu = rect["top"]
            rr = rect["left"] + rect["width"]
            dd = rect["top"] + rect["height"]
        except TypeError as e: # sometimes None values are in the rect
            continue

        area = max(0, min(r, rr) - max(l, ll)) * max(0, min(d, dd) - max(u, uu))
        tags[rect["type_id"] - 1] += area / PATCH_SIZE**2
    
    tags[-1] = 1 - sum(tags)

for screenshot_id, rects in rects_info.items():
    cell ={
        "id": screenshot_id,
        "filename": "%d.png" % (screenshot_id - 1),
        "patches": []
    }
    folder = result_spec["root"] + "%d/" % screenshot_id
    if "%d" % screenshot_id not in os.listdir(result_spec["root"]):
        os.mkdir(result_spec["root"] + "%d/" % screenshot_id)
    if cell["filename"] not in os.listdir(folder):
        shutil.copyfile("raw_images/" + cell["filename"], folder + cell["filename"])
    img = Image.open(folder + cell["filename"])
    cell["width"], cell["height"] = img.size
    for i in range(0, cell["width"], PATCH_SIZE):
        for j in range(0, cell["height"], PATCH_SIZE):
            # lurd
            left, top, right, bottom = i, j, i + PATCH_SIZE - 1, j + PATCH_SIZE - 1
            cropped = img.crop((left, top, right + 1, bottom + 1))
            cropped_filename = "%dx%d.png" % (i//PATCH_SIZE, j//PATCH_SIZE)
            cropped.save(folder + cropped_filename)

            tags = [0] * len(result_spec["tags"])
            calc_tags(tags, rects, (left, top, right, bottom))
            
            cell["patches"].append({
                "filename": cropped_filename,
                "left": i,
                "top": j,
                "right": i + PATCH_SIZE - 1,
                "bottom": j + PATCH_SIZE - 1,
                "tags" : tags
            })

    result_spec["data"].append(cell)

with open("spec.json", "w") as f:
    f.write(json.dumps(result_spec, indent=4))