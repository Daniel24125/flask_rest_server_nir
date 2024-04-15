from ctypes import *

if __name__ == "__main__": 
    print("Starting USB NIR spectrometer connection...")
    
    try: 
        print("Loading the dll file")
        dll = WinDLL("./utils/nir/dll/HSSUSB2A.dll")
        connection = dll.StartMeasure
        print(dll._name)
    except Exception as X:
        print("ERROR: ", X)


