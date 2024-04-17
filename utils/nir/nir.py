from ctypes import *

if __name__ == "__main__": 
    print("Starting USB NIR spectrometer connection...")
    
    try: 
        print("Loading the dll file")
        dll = WinDLL("./utils/nir/dll/Dll1.dll")
        connection = dll.InitSpec()
        print(connection)
    except AttributeError as X:
        print(X)


