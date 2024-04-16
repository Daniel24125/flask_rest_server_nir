from ctypes import *

if __name__ == "__main__": 
    print("Starting USB NIR spectrometer connection...")
    
    try: 
        print("Loading the dll file")
        dll = WinDLL("./utils/nir/dll/HSSUSB2A_Int.dll")
        connection = dll.TestUSBFunc()
        print(connection)
    except AttributeError as X:
        print(X)


