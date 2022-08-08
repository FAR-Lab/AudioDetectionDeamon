#import sys
#sys.path.append("/home/pi/.local/lib/python3.7/site-packages")

import pyaudio
import queue
import threading
import numpy as np

import math
from numpy_ringbuffer import RingBuffer





FORMAT=np.float32

class MicArray(object):

    def __init__(self, rate=16000, channels=6):
        self.pyaudio_instance = pyaudio.PyAudio()
        self.queue = queue.Queue()
        self.quit_event = threading.Event()
        self.channels = channels
        self.sample_rate = rate
        self.chunk_size = 1024
        self.rbuff = RingBuffer(capacity=rate*1, dtype=FORMAT)  #DOA_FRAMES*440 => 128,86 rate => 128,47

        
        

        device_index = None
        for i in range(self.pyaudio_instance.get_device_count()):
            dev = self.pyaudio_instance.get_device_info_by_index(i)
            name = dev['name'].encode('utf-8')
            print(i, name, dev['maxInputChannels'], dev['maxOutputChannels'])
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
            if(b'ReSpeaker 4 Mic Array' in name):
                print('Use {} based on its name!'.format(name),dev)
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
            framesData = np.fromstring(frames, dtype=FORMAT)
            
            self.rbuff.extend(framesData[1::self.channels]) ## here we are picking only one of the 6 channels 
            if(not self.rbuff.is_full):
                continue
            
            if(self.queue.qsize()<1):
                yield True
        print("Left the while loop")
           
    

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
        return np.array(self.rbuff)
    
   
    


    

if __name__ == '__main__':
    pass