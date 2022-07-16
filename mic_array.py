#import sys
#sys.path.append("/home/pi/.local/lib/python3.7/site-packages")

import pyaudio
import queue
import threading
import numpy as np
from gcc_phat import gcc_phat
import math

from numpy_ringbuffer import RingBuffer

import webrtcvad

SOUND_SPEED = 343.2

MIC_DISTANCE_6P1 = 0.064
MAX_TDOA_6P1 = MIC_DISTANCE_6P1 / float(SOUND_SPEED)

MIC_DISTANCE_4 = 0.08127
MAX_TDOA_4 = MIC_DISTANCE_4 / float(SOUND_SPEED)

VAD_FRAMES = 10
DOA_FRAMES = 200  

FORMAT=np.float32

class MicArray(object):

    def __init__(self, rate=16000, channels=8):
        self.pyaudio_instance = pyaudio.PyAudio()
        self.queue = queue.Queue()
        self.quit_event = threading.Event()
        self.channels = channels
        self.sample_rate = rate
        self.chunk_size = rate * VAD_FRAMES / 1000
        self.rbuff = RingBuffer(capacity=rate*channels, dtype=FORMAT)  #DOA_FRAMES*440 => 128,86 rate => 128,47

        
        self.vad = webrtcvad.Vad(3)
        self.speech_count = 0
        #self.doa_chunks=[]
        self.doa_chunks_target_count = int(DOA_FRAMES / VAD_FRAMES)
        self.doa_chunk_count=0
        self.lastDirection=0

        device_index = None
        for i in range(self.pyaudio_instance.get_device_count()):
            dev = self.pyaudio_instance.get_device_info_by_index(i)
            name = dev['name'].encode('utf-8')
            #print(i, name, dev['maxInputChannels'], dev['maxOutputChannels'])
            if(b'USB Lavalier Microphone' in name and channels==1):
                #print('Use {} based on its name!'.format(name))
                device_index = i
                break
            if dev['maxInputChannels'] == self.channels:
                #print('Use {}'.format(name))
                device_index = i
                #break
            if(b'seeed-4mic-voicecard' in name):
                #print('Use {} based on its name!'.format(name))
                device_index = i
                break

        if device_index is None:
            raise Exception('can not find input device with {} channel(s)'.format(self.channels))

        self.stream = self.pyaudio_instance.open(
            input=True,
            start=False,
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=int(self.sample_rate),
            frames_per_buffer=int(self.chunk_size),
            stream_callback=self._callback,
            input_device_index=device_index,
        )

    def _callback(self, in_data, frame_count, time_info, status):
        self.queue.put(in_data)
        return None, pyaudio.paContinue

    def start(self):
        self.queue.queue.clear()
        self.stream.start_stream()


    def read_chunks(self):
        self.quit_event.clear()
        while not self.quit_event.is_set():
            frames = self.queue.get()
            if not frames:
                break
            
            frames = np.fromstring(frames, dtype=FORMAT)
            

            self.rbuff.extend(frames)
            if(not self.rbuff.is_full):
                continue

          #  print(frames.shape,frames.min(),frames.max())

            if self.channels>2:
                if self.vad.is_speech(frames[0::self.channels].tobytes(), self.sample_rate):
                    self.speech_count += 1

            
                self.doa_chunk_count+=1
                if self.doa_chunk_count==self.doa_chunks_target_count:
                    if self.speech_count > (self.doa_chunks_target_count / 2):
                        self.lastDirection =self.get_direction()
                    self.speech_count=0
                    self.doa_chunk_count=0
            
            if(self.queue.qsize()<1):
                yield int(self.lastDirection)
           
    

    def stop(self):
        self.quit_event.set()
        self.stream.stop_stream()
        self.queue.put('')

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        if value:
            return False
        self.stop()

    def get_mono_currentbuff(self):
        return np.array(self.rbuff[::self.channels])
    

    def get_direction(self):
        best_guess = None
        #buf= np.concatenate(self.doa_chunks)
        buf=np.array(self.rbuff)[-12800:]
        #print(buf.shape,self.rbuff.shape)
        if self.channels == 8:
            MIC_GROUP_N = 3
            MIC_GROUP = [[1, 4], [2, 5], [3, 6]]

            tau = [0] * MIC_GROUP_N
            theta = [0] * MIC_GROUP_N

            # buf = np.fromstring(buf, dtype='int16')

            for i, v in enumerate(MIC_GROUP):
                tau[i], _ = gcc_phat(buf[v[0]::8], buf[v[1]::8], fs=self.sample_rate, max_tau=MAX_TDOA_6P1, interp=1)
                theta[i] = math.asin(tau[i] / MAX_TDOA_6P1) * 180 / math.pi

            min_index = np.argmin(np.abs(tau))
            if (min_index != 0 and theta[min_index - 1] >= 0) or (min_index == 0 and theta[MIC_GROUP_N - 1] < 0):
                best_guess = (theta[min_index] + 360) % 360
            else:
                best_guess = (180 - theta[min_index])

            best_guess = (best_guess + 120 + min_index * 60) % 360
        elif self.channels == 4:
            MIC_GROUP_N = 2
            MIC_GROUP = [[0, 2], [1, 3]]

            tau = [0] * MIC_GROUP_N
            theta = [0] * MIC_GROUP_N
            for i, v in enumerate(MIC_GROUP):
                tau[i], _ = gcc_phat(buf[v[0]::4], buf[v[1]::4], fs=self.sample_rate, max_tau=MAX_TDOA_4, interp=1)
                theta[i] = math.asin(tau[i] / MAX_TDOA_4) * 180 / math.pi

            if np.abs(theta[0]) < np.abs(theta[1]):
                if theta[1] > 0:
                    best_guess = (theta[0] + 360) % 360
                else:
                    best_guess = (180 - theta[0])
            else:
                if theta[0] < 0:
                    best_guess = (theta[1] + 360) % 360
                else:
                    best_guess = (180 - theta[1])

                best_guess = (best_guess + 90 + 180) % 360


            best_guess = (-best_guess + 120) % 360

             
        elif self.channels == 2:
            pass

        return best_guess



    

if __name__ == '__main__':
    pass