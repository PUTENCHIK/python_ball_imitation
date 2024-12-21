import cv2


window = cv2.namedWindow("w", cv2.WINDOW_GUI_NORMAL)
# camera = cv2.VideoCapture("http://192.168.254.3:8080")
camera = cv2.VideoCapture("rtsp://192.168.254.3:8080/h264_ulaw.sdp")

while True:
    ret, frame = camera.read()
    
    cv2.imshow("w", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()
