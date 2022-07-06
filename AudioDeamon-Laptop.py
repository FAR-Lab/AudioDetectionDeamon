
import sys

from mic_array import MicArray
import numpy as np
import time
import tensorflow as tf
import tensorflowjs as tfjs
import json

from librosa import power_to_db

import cv2

def main():
    try:
        model = tfjs.converters.load_keras_model("model.json")
        model.compile()
        classes=None
        with open("./metadata.json") as f:
            classes = json.loads(f.read())

        with MicArray(48000, 1)  as mic:
            val=time.time()
            cv2.startWindowThread()
            cv2.namedWindow("preview")
            for chunk in mic.read_chunks():
                
                if(time.time()>val):
                    startEvalTime=time.time()
                    buffer = mic.get_currentbuff(44100)
                    #print("Buffer length should == rate",len(buffer))
                    waveform = tf.cast(buffer, dtype=tf.float32)
                    spectrogram = tf.signal.stft(waveform, frame_length=2048, frame_step=1000)
                    spectrogram = power_to_db(tf.abs(spectrogram)[:,:232]**2)
                    s = spectrogram[tf.newaxis,..., tf.newaxis]

                    #print("original shape",s.shape)

                    #s=mic.get_spectrogram()-80.0
                    #print("original shape",s.shape)
                    
                    #s -= s.min()
                    #s /= 80
                    #s *= 2550
                    
                    #print(s.min(), s.max(),s.shape)

                    min=spectrogram.min()
                    max=spectrogram.max()-min;
                    img = np.uint8((((spectrogram -min)/max)*255))
                    #print(img.min(), img.max(),img.shape)
                    
                    cv2.imshow("preview",img)
                    cv2.waitKey(1)
                    #cv2.imwrite('testImage.png', img)
                    
                    #print(img.min(), img.max(),img.shape)
                    
                    #s=np.swapaxes(s,0,1)



                   # resized = s[:43,:232]*2
                    

                   # min=resized.min()
                   # max=resized.max()-min;
                   # img = ((resized -min)/max)*255

                   # print(s.min(), s.max(),s.shape)
                    #outArray = np.concatenate( [np.array(chunk[:2]),np.array(resized).flatten()],dtype=np.float32)
                    

                    #s = np.expand_dims(s,axis=0)
                 
                    val= model(s)[0]
                    print(classes['wordLabels'][np.argmax(val)])
                    
                    #print(resized.min(), resized.max(),resized.shape)
                    #sys.stdout.buffer.write(outArray.tobytes())
                  
                    sys.stdout.flush()

                    #sys.stderr.write("  "+str(s.min())+":min  max:"+str(s.max()))
                    #sys.stderr.write(str(len(outArray.tobytes())/4))
                    #sys.stderr.flush()
                    
                    diff=time.time()-startEvalTime
                    
                    val =diff+time.time()
                    
                


    except KeyboardInterrupt:
        cv2.imwrite('testImage.png', img)
        cv2.destroyAllWindows()
        pass
        
    


if __name__ == '__main__':
    main()
