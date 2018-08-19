import torch
import numpy as np
from model import Model32, Model64, Model96, Model128
from classifier import PatchClassifier
import os
from skimage import io
import pyaudio

class Sonifier:
    def __init__(self, filename, patch_size=96, use_gpu=True, verbose=False):
        self.classifier = PatchClassifier(use_gpu=use_gpu)
        self.classifier.load('models/model96.81.83.pt')
        self.patch_size = patch_size

        self.image = np.moveaxis(io.imread(filename), -1, 0) # pull the rgb axis to the front
        self.sonify(verbose)


    def sonify(self, verbose=False):
        tags = [0] * 6
        height, width = self.image.shape[1:3]
        verbose_temp = 0.1
        for i in range(0, height-self.patch_size, self.patch_size):
            for j in range(0, width-self.patch_size, self.patch_size):
                self.image[3, i+0, j+0] = int(j/width*100) # normalized left (0 - 100)
                self.image[3, i+0, j+1] = int(i/height*100) # normalized top (0 - 100)
                self.image[3, i+0, j+2] = abs(50-self.image[3, 0, 0]) * 2 # normalized min distance from either left or right edges (0 - 100), e.g., abs(50 - norm_left) * 2
                self.image[3, i+0, j+3] = abs(50-self.image[3, 0, 1]) * 2 # normalized min distance from either top or bottom edges (0 - 100), e.g., abs(50 - norm_top) * 2
                scores = self.classifier.classify_one(self.image[:,i:i+self.patch_size,j:j+self.patch_size])
                tags[np.argmax(scores)] += 1

                if verbose and verbose_temp < (width*i+j)/height/width:
                    print(int(verbose_temp*100), "%")
                    verbose_temp += 0.1
        self.tags = {
            "text": tags[0] / sum(tags),
            "image": tags[1] / sum(tags),
            "graph": tags[2] / sum(tags),
            "ad": tags[3] / sum(tags),
            }
        if verbose:
            print(self.tags)

    def play(self):
        scale = min(10, self.image.shape[1] / 500)
        p_ad = self.tags["ad"] * scale
        p_image = (self.tags["image"] + self.tags["graph"]) * scale
        p_text = self.tags["text"] * scale

        s_text = [261.63]
        s_image = [293.66, 392.00]
        s_ad = [493.88, 523.25, 587.33]

        freqs_dura_list = list(zip(
            [s_text, s_image, s_ad],
            [p_text * 3, p_image * 3, p_ad * 3]
        )   )

        self.play_sound(freqs_dura_list)

    def play_sound(self, freqs_dura_list, vol=0.5):
        p = pyaudio.PyAudio()
        volume = vol     # range [0.0, 1.0]
        fs = 44100       # sampling rate, Hz, must be integer
        samples = []

        past_sample_length = 0
        for freqs, duration in freqs_dura_list:
            # generate samples, note conversion to float32 array
            sample_length = int(fs*duration)
            base = np.arange(past_sample_length, past_sample_length + sample_length)
            #base = np.arange(sample_length)
            data = np.zeros(sample_length)
            past_sample_length += sample_length
            for f in freqs:
                data += np.sin(2*np.pi*base*f/fs)
            data /= len(freqs)
            samples.append(data.astype(np.float32))

        # for paFloat32 sample values must be in range [-1.0, 1.0]
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=fs,
                        output=True)

        # play. May repeat with different volume values (if done interactively) 
        for sample in samples:
            stream.write(volume * sample)
        #stream.write(volume*np.concatenate(samples).astype(np.float32))

        stream.stop_stream()
        stream.close()

        p.terminate()


if __name__ == "__main__":
    for i in range(1,9):
        Sonifier('samples/%d.png' % i, use_gpu=False, verbose=True).play()
