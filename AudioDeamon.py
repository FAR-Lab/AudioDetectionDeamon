
import sys

from mic_array import MicArray
import numpy as np
import time

import cv2

def main():
    try:
        with MicArray(22050, 4)  as mic:
            val=time.time()
            
            for chunk in mic.read_chunks():
                
                if(time.time()>val):
                    s=mic.get_spectrogram()
                    
                    s -= s.min()
                    s /= 80
                    s *= 255
                    
                    s=np.swapaxes(s,0,1)
                    resized = cv2.resize(s, (232,43), interpolation = cv2.INTER_CUBIC)
                    outArray = np.concatenate( [np.array(chunk[:2]),np.array(resized).flatten()],dtype=np.float32)
                    
                    #cv2.imwrite('images/savedimage.png', resized)
                    #print(resized.min(), resized.max(),resized.shape,s.shape)
                    sys.stdout.buffer.write(outArray.tobytes())
                  
                    sys.stdout.flush()

                    #sys.stderr.write("  "+str(s.min())+":min  max:"+str(s.max()))
                    #sys.stderr.write(str(len(outArray.tobytes())/4))
                    #sys.stderr.flush()
                    
                    val =0.5+time.time()
                    
                


    except KeyboardInterrupt:
        pass
        
    


if __name__ == '__main__':
    main()
