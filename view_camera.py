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

def dupdate(value):
    global delta
    delta = value


def aupdate(value):
    global amount
    amount = value


main_window = "Window"
debug_window = "Changed"
window = cv2.namedWindow(main_window, cv2.WINDOW_GUI_NORMAL)
window = cv2.namedWindow(debug_window, cv2.WINDOW_GUI_NORMAL)
camera = cv2.VideoCapture("rtsp://192.168.254.3:8080/h264_ulaw.sdp")

cv2.setMouseCallback(main_window, on_mouse_callback)
ball_position = None

low, up = 44, 65
cv2.createTrackbar("Lower", main_window, low, 255, lupdate)
cv2.createTrackbar("Upper", main_window, up, 255, uupdate)

delta = 35
cv2.createTrackbar("Delta", debug_window, delta, 255, dupdate)
amount = 5
cv2.createTrackbar("Amount", debug_window, amount, 255, aupdate)

while True:
    _, frame = camera.read()
    origin = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if ball_position:
        cv2.circle(origin,
                   (ball_position[1], ball_position[0]),
                   5, (255, 255, 0), 2)
        
        pixel = hsv[ball_position[0], ball_position[1]]
        cv2.putText(origin,
                f"{pixel}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3)
        
        lower = np.array([pixel[0]-delta, pixel[1]-delta, pixel[2]-delta])
        upper = np.array([pixel[0]+delta, pixel[1]+delta, pixel[2]+delta])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.dilate(mask, np.ones((10, 10)))
        thresh = cv2.bitwise_and(frame, frame, mask=mask)
        
        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        _, result = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        big_contours = dict()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10_000:
                big_contours[area] = contour
                
        for contour_area in sorted(big_contours)[-amount-1:]:
            contour = big_contours[contour_area]
            
            eps = 1e-3 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, eps, True)
            for p in approx:
                cv2.circle(origin, tuple(*p), 2, (255, 0, 0), 2)
                
            cv2.drawContours(origin, [contour], 0, (0, 255, 0), 3)
        
        cv2.imshow(debug_window, result)
    
    # lower = np.array([0, low, 0])
    # upper = np.array([255, up, 255])
    # mask = cv2.inRange(hsv, lower, upper)
    # mask = cv2.dilate(mask, np.ones((10, 10)))

    # result = cv2.bitwise_and(frame, frame, mask=mask)
    
    cv2.imshow(main_window, origin)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()
