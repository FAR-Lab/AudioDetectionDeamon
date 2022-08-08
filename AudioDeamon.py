
from audioop import rms
import sys

from mic_array import MicArray
import numpy as np
import time
import tensorflow as tf
import tensorflowjs as tfjs
import json

from librosa import power_to_db, resample
from numpy_ringbuffer import RingBuffer


import cv2

def main():
    dispaly = (len(sys.argv)>1 and sys.argv[1]=="display")
    avgBuff = RingBuffer(capacity= 15, dtype='1f')

    try:
        model = tfjs.converters.load_keras_model("/home/listeningbot/customNodes/node-red-contrib-ml-select/modelFolder/model.json")
        model.compile()
        classes=None
        with open("/home/listeningbot/customNodes/node-red-contrib-ml-select/modelFolder/metadata.json") as f:
            classes = json.loads(f.read())

        with MicArray(48000, 2)  as mic:
            val=time.time()


            if(dispaly):
                #cv2.startWindowThread()
                #cv2.namedWindow("preview")
                pass

            for chunk in mic.read_chunks():

                if(time.time()>val):
                    
                    ### Get the Buffer
                    buffer = resample(mic.get_mono_currentbuff(),orig_sr=mic.sample_rate,target_sr=44100)

                    ### Create the spectrogram
                    waveform = tf.cast(buffer, dtype=tf.float32)
                    spectrogram = tf.signal.stft(waveform, frame_length=2048, frame_step=1000)
                    spectrogram = power_to_db(tf.abs(spectrogram)[:,:232]**2,ref=np.median)
                    s = spectrogram[tf.newaxis,..., tf.newaxis]

                    ### Compute the Prediction
                    output= model(s)[0]
                    cls = classes['wordLabels'][np.argmax(output)]

                    ### Compute the current volume
                    volume = np.rint(np.sqrt(np.mean(buffer**2))*10000)
                    avgBuff.append(volume)
                    
                    avgVolume = get_cur_avg(avgBuff)
                    print("Average Volume")
                    print(avgVolume)
                    ### write output to the stdout
                    sys.stdout.write(str(cls)+','+str(volume)+','+str(chunk)+'\r\n')
                    sys.stdout.flush()

                    if(dispaly):
                        min=spectrogram.min()
                        max=spectrogram.max()-min;
                        img = np.uint8((((spectrogram -min)/max)*255))
                        dim=(img.shape[1]*4,img.shape[0]*4)
                        cv2.putText(img=img, text=cls, org=(10, 10), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)
                        cv2.imwrite("tmp.png",img)
                        #cv2.imshow("preview",cv2.resize(img, dim))
                        #cv2.waitKey(1)




                    #sys.stderr.write("  "+str(s.min())+":min  max:"+str(s.max()))
                    #sys.stderr.write(str(len(outArray.tobytes())/4))
                    #sys.stderr.flush()
                    val =0.5+time.time()




    except KeyboardInterrupt:
        if dispaly :
            cv2.imwrite('testImage.png', img)
            cv2.destroyAllWindows()
        pass

def get_cur_avg(avgBuff):
    sum = 0
    count = 0
    print(avgBuff)
    for i in range(len(avgBuff)):
        vol1 = avgBuff[i-1]
        vol2 = avgBuff[i] 
        diff = vol2 - vol1
        sum += diff
        count += 1

    return sum / count



if __name__ == '__main__':
    main()
