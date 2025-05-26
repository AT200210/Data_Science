import cv2

class FrameDrawing:
# Draw text into the frame
    @staticmethod
    def draw_text(frame, text, org, font, font_scale, color, thickness, line_type=None, bottom_left_origin=None):
# Returns the text in the frame
        return cv2.putText(frame, text, org, font, font_scale, color, thickness, line_type, bottom_left_origin)
    
#  Draw rectangle
    @staticmethod
    def draw_rect(frame, center_position, radius, color, line_thickness):

        return cv2.rectangle(frame, center_position, radius, color, line_thickness)

# Draw circle
    @staticmethod
    def draw_circle(frame, center_position, radius, color, line_thickness):

        return cv2.circle(frame, center_position, radius, color, line_thickness)

# Draw ellipse
    @staticmethod
    def draw_ellipse(frame, center_coordinates, axes_length, angle, start_angle, end_angle ,color, thickness):

        return cv2.ellipse(frame, center_coordinates, axes_length, angle, start_angle, end_angle ,color, thickness)
