import cv2
from config.frame_drawing import FrameDrawing
from config.frame_editing import FrameEditing

# A class to find the width and height of teh frame
class FindValues:
    # Get frame width and height
    @staticmethod
    def get_frame_width_height(frame):

        return frame.shape[1], frame.shape[0]
    
# A class that contain the face detection algorithm
class FaceDetection:

    @staticmethod
    def face_detection(frame, face_cascade, scale_factor, min_neighbors, line_thickness):
        width,height=FindValues.get_frame_width_height(frame)

        gray=FrameEditing.convert_frame_to_gray(frame)

        faces=face_cascade.detectMultiScale(gray, scale_factor, min_neighbors, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x,y,w,h) in faces:

            FrameDrawing.draw_rect(frame, (x,y), (x+w, y+h), (255,53,18), line_thickness)

            print(f'Coordinate: {(x,y)} - Size: {(w,h)}')

            FrameDrawing.draw_text(frame, 'Coordinates: {} - Size: {}'.format((x,y), (w,h)),(width//25, height//10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)

class LiveFaceDetection:
    def __init__(self):
        self.live_face_detection()


    @classmethod
    def live_face_detection(cls):

        cap=cv2.VideoCapture(0)

        print('Camera On')

        face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        while True:

            ret, frame=cap.read()

            small_frame=FrameEditing.scale_frame(frame, 0.5, 0.5)

            FaceDetection.face_detection(small_frame, face_cascade, scale_factor=1.2, min_neighbors=5, line_thickness=2)

            cv2.imshow('Vide Frame', small_frame)

            if cv2.waitKey(1)==ord('q'):
                print("Camera off")
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__=='__main__':
    face_detection=LiveFaceDetection()