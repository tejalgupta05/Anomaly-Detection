import cv2
 

"""
# from webcam
cap = cv2.VideoCapture(0)
i = 1

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if i > 100:
        break
    
    # Save Frame by Frame into disk using imwrite method
    cv2.imwrite('Frame'+str(i)+'.jpg', frame)
    i += 1

cap.release()
cv2.destroyAllWindows()
"""


# from video stream
cap = cv2.VideoCapture(r"D:\ML\Mini Project\1.2.mp4")
i = 1

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == False:
        break
    
    # Save Frame by Frame into disk using imwrite method
    cv2.imwrite(f"D:\\ML\\Mini Project\\images\\1\\Frame {i}.jpg", frame)
    i += 1

cap.release()
cv2.destroyAllWindows()