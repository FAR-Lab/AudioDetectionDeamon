
import numpy as np
from mic_array import MicArray



def main():
    try:
        with MicArray(16000, 4)  as mic:
            for chunk in mic.read_chunks():
                print(chunk[0],int(chunk[1]),chunk[2].shape)


    except KeyboardInterrupt:
        pass
        
    


if __name__ == '__main__':
    main()
