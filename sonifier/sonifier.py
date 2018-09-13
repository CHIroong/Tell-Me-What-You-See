import os
import json

from gtts import gTTS
from pydub import AudioSegment

from classifier.classifier import PatchClassifier
from sonifier.sonified_result import SonifiedResult

class Sonifier:
    def __init__(self, spec_filename, patch_size=96, use_gpu=True, verbose=False):
        self.patch_size = patch_size
        with open(spec_filename, "r") as f:
            self.spec = json.loads(f.read())
        self.verbose = verbose
        self.tags = []
        self.doc_features = []

        root = self.spec["root"]
        for data in self.spec["data"]:
            self.doc_features.append(data["doc_features"])
            folder = root + ('%s/' % data["id"])
            width, height = data["width"], data["height"]
            tags = [0] * 6
            verbose_temp = 0.1
            for patch in data["patches"]:
                tags = patch["tags"]
                self.tags.append({
                    "text": tags[0] / sum(tags),
                    "image": tags[1] / sum(tags),
                    "graph": tags[2] / sum(tags),
                    "ad": tags[3] / sum(tags),
                    })

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
        sentence = " ".join(self.doc_features[data_id]["keywords"])
        self.save_tts_mp3(sentence, output_filename, 1.5)

    def generate_summary_mp3(self, data_id, output_filename):
        doc_feature = self.doc_features[data_id]
        num_imgs = int(doc_feature["num_imgs"])
        num_links = int(doc_feature["num_links"])
        num_headers = int(doc_feature["num_headers"])
        num_tables = int(doc_feature["num_tables"])
        sentence = f"헤더 {num_headers}개 링크 {num_links}개 이미지 {num_imgs}개 표 {num_tables}개"
        self.save_tts_mp3(sentence, output_filename, 1.5)
    
    def generate_defaults_mp3(self, data_id, output_filename):
        doc_feature = self.doc_features[data_id]
        title = doc_feature['title']
        navs = " ".join(doc_feature['navs']) if len(doc_features['navs']) > 0 else "이 없습니다."
        headers = " ".join(doc_feature['headers'][:5]) if len(doc_feature['headers']) > 0 else "가 없습니다."
        sentence = f"페이지 제목 {title} 네비게이션 {navs} 헤더 {headers}"
        self.save_tts_mp3(sentence, output_filename, 1.5)

    def generate_merged_sg_mp3(self, glance_filename, keyword_filename, default_filename, output_filename):
        glance = AudioSegment.from_wav(glance_filename)
        keyword = AudioSegment.from_mp3(keyword_filename)
        default = AudioSegment.from_mp3(default_filename)
        (glance + keyword + default)[:15*1000].export(output_filename, format="mp3")

    def generate_merged_ss_mp3(self, summary_filename, default_filename, output_filename):
        summary = AudioSegment.from_mp3(summary_filename)
        default = AudioSegment.from_mp3(default_filename)
        (summary + default)[:15*1000].export(output_filename, format="mp3")

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
    
    def save_tts_mp3(self, sentence, output_filename, speed=1):
        tts = gTTS(sentence, lang='ko')
        tts.save(output_filename)
        if speed != 1:
            reopened = AudioSegment.from_mp3(output_filename)
            self.speed_change(reopened, speed=1.3).export(output_filename, format="mp3")
