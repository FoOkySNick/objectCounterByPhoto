import cv2

cap = cv2.VideoCapture(0)
cap.set(3,640)  # set Width
cap.set(4,480)  # set Height
ret, image = cap.read()
cv2.imwrite("input.jpg", image)
