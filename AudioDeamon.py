from audioop import rms
import sys

from mic_array import MicArray
import numpy as np
import time
import tensorflow as tf
import tensorflowjs as tfjs
import json


from pathlib import Path

from tuning import Tuning
import usb.core
import time

from librosa import power_to_db, resample 

from numpy_ringbuffer import RingBuffer

import cv2

def main():
    UPDATE_RATE=1.0
    avgBuff = RingBuffer(capacity=int(20/UPDATE_RATE), dtype=np.float32)
    dispaly = (len(sys.argv)>1 and sys.argv[1]=="display")

    dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    Mic_tuning = Tuning(dev)
    #Mic_tuning.write('AGCONOFF',0)
    try:
        parentDir = Path(__file__).absolute().parents[1]
        model = tfjs.converters.load_keras_model(Path(parentDir,"modelFolder/model.json"))
        model.compile()
        classes=None  
        with open(Path(parentDir,"modelFolder/metadata.json")) as f:
            classes = json.loads(f.read())
        direction=366
        with MicArray(16000, 2)  as mic:

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
                    #print("Got the S")
                    ### Compute the Prediction
                    output= model(s)[0]

                    clas = classes['wordLabels'][np.argmax(output)]
                    
                    ### Compute the current volume
                    volume = np.rint(np.sqrt(np.mean(buffer**2))*10000)
                    avgBuff.append(volume)
                    volumneSlow=volume
                    volumechange = 0.0
                    if avgBuff.is_full:
                        length = int(np.round(avgBuff.maxlen/2))
                        vnew = np.array(avgBuff)[length:].mean()
                        vold = np.array(avgBuff)[:avgBuff.maxlen-length].mean()
                        volumechange =vnew-vold
                       
                        volumneSlow = np.array(avgBuff).mean()
                   
                    ### write output to the stdout
                    sys.stdout.write(str(clas)+','+str(volumneSlow)+','+str(direction)+','+ str(volumechange) +'\r\n')
                    #print("About to write")
                    #sys.stdout.write(str(clas)+','+str(volume)+','+str(Mic_tuning.direction)+'\r\n')
                    #print("Finished writing")
                    sys.stdout.flush()

                    if(dispaly):
                        min=spectrogram.min()
                        max=spectrogram.max()-min
                        img = np.uint8((((spectrogram -min)/max)*255))
                        dim=(img.shape[1]*4,img.shape[0]*4)
                        cv2.putText(img=img, text=clas, org=(10, 10), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)
                        cv2.imwrite("tmp.png",img)
                        #cv2.imshow("preview",cv2.resize(img, dim))
                        #cv2.waitKey(1)




                    #sys.stderr.write("  "+str(s.min())+":min  max:"+str(s.max()))
                    #sys.stderr.write(str(len(outArray.tobytes())/4))
                    #sys.stderr.flush()

                    val =UPDATE_RATE+time.time()
                else:
                    try:
                        direction=Mic_tuning.direction
                        time.sleep(0.1)
                        print("Got direction")
                    except usb.core.USBError as e:
                        print(e)
                        pass
                    
                
    except KeyboardInterrupt:
        if dispaly :
            cv2.imwrite('testImage.png', img)
            cv2.destroyAllWindows()
        else:
            mic.stop()
            exit()
        pass



if __name__ == '__main__':
    main()
