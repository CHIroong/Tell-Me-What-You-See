import torch
import numpy as np
from model import Model32, Model64, Model96, Model128
from classifier import PatchClassifier
import os
import json
from skimage import io
from sonified_result import SonifiedResult
from gtts import gTTS
from pydub import AudioSegment

class Sonifier:
    def __init__(self, spec_filename, patch_size=96, use_gpu=True, verbose=False):
        self.classifier = PatchClassifier(use_gpu=use_gpu)
        self.classifier.load('models/model96.87.47.pt')
        self.patch_size = patch_size
        with open(spec_filename, "r") as f:
            self.spec = json.loads(f.read())
        self.verbose = verbose
        self.tags = []
        self.keywords = []


    def sonify(self):
        root = self.spec["root"]
        for data in self.spec["data"]:
            folder = root + ('%s/' % data["id"])
            width, height = data["width"], data["height"]
            tags = [0] * 6
            verbose_temp = 0.1
            vis = io.imread(folder + data["filename"])
            for patch in data["patches"]:
                patch_filename = folder + patch["filename"]
                patch_image = np.moveaxis(io.imread(patch_filename), -1, 0)
                patch_image[3, 0, 0] = patch["left"] / width * 100
                patch_image[3, 0, 1] = patch["top"] / height * 100
                patch_image[3, 0, 2] = abs(50-patch_image[3, 0, 0]) * 2 # normalized min distance from either left or right edges (0 - 100), e.g., abs(50 - norm_left) * 2
                patch_image[3, 0, 3] = abs(50-patch_image[3, 0, 1]) * 2 # normalized min distance from either top or bottom edges (0 - 100), e.g., abs(50 - norm_top) * 2
                patch_image[3, 0, 4] = float(patch["features"]["cursor"] == "pointer")
                ratio = patch["features"]["aspect_ratio"]
                if ratio >= 255:
                    ratio = 255
                patch_image[3, 0, 5] = ratio
                patch_image[3, 0, 6] = float(patch["features"]["is_img"])
                patch_image[3, 0, 7] = float(patch["features"]["is_iframe"])
                patch_image[3, 0, 8] = patch["features"]["nested_a_tags"]
                patch_image[3, 0, 9] = float(patch["features"]["contains_harmful_url"])
                scores = self.classifier.classify_one(patch_image)
                tags[np.argmax(scores)] += 1

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

            self.tags.append({
                "text": tags[0] / sum(tags),
                "image": tags[1] / sum(tags),
                "graph": tags[2] / sum(tags),
                "ad": tags[3] / sum(tags),
                })
            self.keywords.append(data["keywords"])
            if self.verbose:
                print(self.tags)
                io.imsave('_'.join((folder + data["filename"]).split('.')[:-1]) + '_vis.png', vis)

    def generate_glance_wav(self, data_id, output_filename):
        scale = min(5, self.spec["data"][data_id]["height"] / 1500)
        duration_text = self.tags[data_id]["text"] * scale
        duration_image = (self.tags[data_id]["image"] + self.tags[data_id]["graph"]) * scale
        duration_ad = self.tags[data_id]["ad"] * scale
        
        freqs_text = [261.63]
        freqs_image = [293.66, 392.00]
        freqs_ad = [493.88, 523.25, 587.33]

        sr = SonifiedResult()
        sr.append_sinewave(freqs_text, duration_milliseconds=duration_text * 1000) #volume can be also adjusted by volume=0.0~1.0
        #sr.append_silence(duration_milliseconds=500)
        sr.append_sinewave(freqs_image, duration_milliseconds=duration_image * 1000) #volume can be also adjusted by volume=0.0~1.0
        #sr.append_silence(duration_milliseconds=500)
        sr.append_sinewave(freqs_ad, duration_milliseconds=duration_ad * 1000) #volume can be also adjusted by volume=0.0~1.0
        sr.append_silence(duration_milliseconds=200)
        sr.save_wav(output_filename)
    
    def generate_keywords_mp3(self, data_id, output_filename):
        sentence = " ".join(self.keywords[data_id])
        tts = gTTS(sentence, lang='ko')
        tts.save(output_filename)
    
    def generate_merged_mp3(self, glance_filename, keyword_filename, output_filename):
        glance = AudioSegment.from_wav(glance_filename)
        keyword = AudioSegment.from_mp3(keyword_filename)
        merged = glance + self.speed_change(keyword, speed=1.8)
        merged.export(output_filename, format="mp3")

    @staticmethod
    def speed_change(sound, speed=1.0):
        # Manually override the frame_rate. This tells the computer how many
        # samples to play per second
        sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
            "frame_rate": int(sound.frame_rate * speed)
        })
        # convert the sound with altered frame rate to a standard frame rate
        # so that regular playback programs will work right. They often only
        # know how to play audio at standard frame rate (like 44.1k)
        return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

if __name__ == "__main__":
    sonifier = Sonifier('sample_spec.json', use_gpu=False, verbose=True)
    sonifier.sonify()
    for i in range(3):
        sonifier.generate_glance_wav(i, "glance_%d.wav" % i)
        sonifier.generate_keywords_mp3(i, "keyword_%d.mp3" % i)
        sonifier.generate_merged_mp3("glance_%d.wav" % i, "keyword_%d.mp3" % i, "merged_%d.mp3" % i)