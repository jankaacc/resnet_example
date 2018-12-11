import cv2
import numpy as np
from PIL import Image


class VideoCaptureManager():
    
    def __init__(self, device_nr, output_size, cnn_model):
        self.device_nr = device_nr
        self.output_size = output_size
        self.cnn_model = cnn_model
        
    def __enter__(self):
         self.cap = VideoCaptureRecognizer(self.device_nr, self.cnn_model)
         
         if (self.cap.isOpened() == False): 
            print("Unable to read camera feed")
            return -1
        
         return self.cap
        
    def __exit__(self, exctype, value, tb):
        self.cap.release()
        cv2.destroyAllWindows()
        return False
    
        
class VideoCaptureRecognizer(cv2.VideoCapture):

    def __init__(self, device_nr, cnn_model):
        super(VideoCaptureRecognizer, self).__init__(device_nr)
        self.cnn_model = cnn_model
        
    def capture(self):
        while(True):
            ret, frame = self.read() 
            if ret:
                image = Image.fromarray(frame, 'RGB')
                image = image.resize((224,224))
                img_array = np.array(image)
                img_array = np.expand_dims(img_array, axis=0)
                self.cnn_model.make_prediction(img_array,
                                               verbouse_level=1,
                                               top=1)
                cv2.imshow('frame', frame)            
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        
