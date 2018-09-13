import math
import wave
import struct
import numpy as np

class SonifiedResult:

    # Audio will contain a long list of samples (i.e. floating point numbers describing the
    # waveform).  If you were working with a very long sound you'd want to stream this to
    # disk instead of buffering it all in memory list this.  But most sounds will fit in 
    # memory.

    def __init__(self):
        self.audio = []
        self.sample_rate = 44100.0

    def append_silence(self, duration_milliseconds=500):
        """
        Adding silence is easy - we add zeros to the end of our array
        """
        num_samples = duration_milliseconds * (self.sample_rate / 1000.0)

        for x in range(int(num_samples)): 
            self.audio.append(0.0)

        return


    def append_sinewave(self, freqs, duration_milliseconds=500, volume=1.0):
        """
        The sine wave generated here is the standard beep.  If you want something
        more aggresive you could try a square or saw tooth waveform.   Though there
        are some rather complicated issues with making high quality square and
        sawtooth waves... which we won't address here :) 
        """ 

        num_samples = duration_milliseconds * (self.sample_rate / 1000.0)

        for x in range(int(num_samples)):
            self.audio.append(volume * np.mean([math.sin(2 * math.pi * freq * ( x / self.sample_rate )) for freq in freqs]))

        return


    def save_wav(self, file_name):
        # Open up a wav file
        wav_file=wave.open(file_name,"w")

        # wav params
        nchannels = 1

        sampwidth = 2

        # 44100 is the industry standard sample rate - CD quality.  If you need to
        # save on file size you can adjust it downwards. The stanard for low quality
        # is 8000 or 8kHz.
        nframes = len(self.audio)
        comptype = "NONE"
        compname = "not compressed"
        wav_file.setparams((nchannels, sampwidth, self.sample_rate, nframes, comptype, compname))

        # WAV files here are using short, 16 bit, signed integers for the 
        # sample size.  So we multiply the floating point data we have by 32767, the
        # maximum value for a short integer.  NOTE: It is theortically possible to
        # use the floating point -1.0 to 1.0 data directly in a WAV file but not
        # obvious how to do that using the wave module in python.
        for sample in self.audio:
            wav_file.writeframes(struct.pack('h', int( sample * 32767.0 )))

        wav_file.close()

        return

if __name__ == "__main__":

    sr = SonifiedResult()

    sr.append_sinewave([403], duration_milliseconds=1000, volume=0.25)
    sr.append_silence(duration_milliseconds=100)
    sr.append_sinewave([403, 450], volume=0.5)
    sr.append_silence(duration_milliseconds=100)
    sr.append_sinewave([403, 600, 800])
    sr.save_wav("output.wav")

"""
import pyaudio
import numpy as np
import wave
import struct

def sample(freqs_dura_list, vol=0.5):
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

        p.terminate()

        wav_file = wave.open("helpme.wav", "w")
        nframes = len(samples[0])
        comptype = "NONE"
        compname = "not compressed"
        wav_file.setparams(())

        return stream







stream = sample([([403], 2)])
help(stream._stream)
"""