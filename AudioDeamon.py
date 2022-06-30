
import sys
import numpy as np
from mic_array import MicArray



def main():
    try:
        with MicArray(16000, 4)  as mic:
            for chunk in mic.read_chunks():
                outArray=None
                if(len(chunk)==2):
                    outArray = np.array(chunk[:2],dtype=np.float32)
                else:
                    outArray = np.concatenate( [np.array(chunk[:2]),np.array(chunk[2]).flatten()],dtype=np.float32)
                sys.stdout.buffer.write(outArray.tobytes())
                sys.stdout.flush()


    except KeyboardInterrupt:
        pass
        
    


if __name__ == '__main__':
    main()
