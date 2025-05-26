import cv2
import numpy as np

class FrameEditing:

    @staticmethod
    def scale_frame(frame, f_x, f_y):
        # Return smaller frame
        return cv2.resize(frame, (0,0), fx=f_x, fy=f_y)
    
    @staticmethod
    def convert_frame_to_hsv(frame):
        # Returns HSV color frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    @staticmethod
    def convert_frame_to_gray(frame):
        # Returns gray frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Show only the skin color
    @staticmethod
    def show_skin_color(frame, hsv_frame, lower_color_range, upper_color_range):
# Make a mask with the color that we want to show -> get only 0 or 1 as result
        mask=cv2.inRange(hsv_frame, np.array(lower_color_range), np.array(upper_color_range))

        # Return result
        return cv2.bitwise_and(frame, frame, mask=mask)
    
    @staticmethod
    def combine_two_frame(frame1, frame2):
        return np.concatenate((frame1, frame2), axis=1)
    
    @staticmethod
    def combine_four_frame(original_frame, width, height, frame1, frame2, frame3, frame4):

        frame=np.zeros(original_frame.shape, np.unit8)

        frame[:height//2, :width//2]=frame1 # Top left corner
        frame[height//2:, :width//2]=frame2 # Bottom left corner
        frame[:height//2, width//2:]=frame3  #Top right corner
        frame[height//2:, width//2]=frame[:,:,None] # bottom right corner- adding a 3 dimension in case of a grayscale image

        return frame
    
    