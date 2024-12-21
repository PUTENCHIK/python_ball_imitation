import cv2
import numpy as np


def on_mouse_callback(event, x, y, *params):
    global ball_position
    if event == cv2.EVENT_LBUTTONDOWN:
        ball_position = [y, x]
        
        
def lupdate(value):
    global low
    low = value
    
def uupdate(value):
    global up
    up = value


window_name = "Window"
window = cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
camera = cv2.VideoCapture("rtsp://192.168.254.3:8080/h264_ulaw.sdp")

cv2.setMouseCallback(window_name, on_mouse_callback)
ball_position = None

low, up = 125, 160
cv2.createTrackbar("Lower", window_name, low, 255, lupdate)
cv2.createTrackbar("Upper", window_name, up, 255, uupdate)

while True:
    _, frame = camera.read()
    origin = frame.copy()
    
    if ball_position:
        cv2.circle(frame,
                   (ball_position[1], ball_position[0]),
                   5, (255, 255, 0), 2)
    
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # _, thresh = cv2.threshold(gray, slimit, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(thresh,
    #                                cv2.RETR_TREE,
    #                                cv2.CHAIN_APPROX_SIMPLE)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, low, 0])
    upper = np.array([255, up, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.dilate(mask, np.ones((10, 10)))

    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    cv2.imshow(window_name, result)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()
