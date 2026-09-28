import cv2
import numpy as np

#Change this depending on how far you want the lane to go, further -> lower
detection_range = (1/2)

mask_polygon= np.array([
#Triangle. (left point), (Right point), (Top point). Height stands for bottom y value, Y axis is inversed
[(0, 325), (130, 360), (370, 192), (500, 360), (640, 325), (370, 145)]
])

#This controls how much blur is applied to the image. Higher->More blur
blur_amount = 2

#These control the thresholds of the canny function. Threshold to be adjusted depending on the amount of noise in the video
canny_threshold_1 = 50
canny_threshold_2 = 150

#This controls the threshold of the HoughLinesP function. Threshold to be adjusted if the program isnt able to stay consistant
# with detecting the lanes
houghlines_threshold = 150

def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.001, 0
    y1 = image.shape[0]
    y2 = int(y1 * (detection_range))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2),(y1, y2), 1)
        slope = parameters[0]
        intercept = parameters [1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def canny(lane_image):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = np.copy(gray)
    #Change this if the picture is too noisy / there are too many edges
    for x in range(blur_amount):
        blur = cv2.GaussianBlur(blur, (5, 5), 0)
    canny = cv2.Canny(blur, canny_threshold_1, canny_threshold_2)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = (line.reshape(4))
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_interest(image):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, mask_polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def videoCapture(cap):
    while(cap.isOpened()):
        ret, frame = cap.read()
        edge_image = canny(frame)
        masked_image = region_of_interest(edge_image)
        lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, houghlines_threshold, np.array([]), minLineLength=40, maxLineGap=5)
        if lines is None:
            is_obstructed = True
            line_image = frame
        else:
            is_obstructed = False
            averaged_lines = average_slope_intercept(frame, lines)
            line_image = display_lines(frame, averaged_lines)
        return line_image, is_obstructed  
