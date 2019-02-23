import cv2
import contourRecognition

if __name__ == '__main__':
    recogniser = ConturRecognition.Recogniser('input_one.jpg', 0)
    area = cv2.contourArea(recogniser.findContours())
    f = open('./single_object_area.txt', 'w')
    f.write(area)
    f.close()
