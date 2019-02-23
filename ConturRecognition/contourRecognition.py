import cv2
import os


def write_to_file(data, file):
    f = open(file, 'a', encoding='utf8')
    f.writelines(data)
    f.close()


class Recogniser:
    def __init__(self, path, index):
        self.file = './output.txt'
        self.image = cv2.imread("./input/" + path)
        self.index = index

    def findContours(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    # загрузите изображение, смените цвет на оттенки серого и уменьшите резкость
    # Необходимо выбрать параметры фильтра, обеспечивающие наилучшее подавление шума.
    # Влияние пикселей друг на друга при гауссовой фильтрации обратно пропорционально квадрату расстояния между ними:
    # коэффициент пропорциональности, а, следовательно, и степень размытия, определяются параметром σ.
    # Чрезмерное повышение коэффициента приведёт к усилению усреднения вплоть до равномерно чёрного цвета всех пикселей:
    # gray = cv2.GaussianBlur(gray, (3, 3), 10)
        gray = cv2.GaussianBlur(gray, (7, 7), 1)
        edged = cv2.Canny(gray, 3, 250, apertureSize=5, L2gradient=True)
        cv2.imwrite("./output_photos/edged{0}.jpg".format(str(self.index)), edged)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        return cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    def count(self, data):
        quantity = Recogniser.count_by_area(self, self.image, data)
        write_to_file("Я нашёл {0} объектов на этой картинке\n".format(quantity), self.file)

        return quantity

    def count_by_area(self, image, data):
        images, contours, ret = data
        area = cv2.contourArea(contours[-1])
        holes_area = 0
        single_area = 0
        f = open('./single_object_area.txt', 'r')
        for line in f:
            single_area = float(line.strip())

        for c in contours:
            c_area = cv2.contourArea(c)
            if c_area < area / 15:
                holes_area += c_area
            # аппроксимируем (сглаживаем) контур
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)

        area = area - holes_area
        cv2.imwrite("./output_photos/output{0}.jpg".format(str(self.index)), image)

        return round(area / single_area)


if __name__ == "__main__":
    photos = os.listdir('./input')
    open('./output.txt', 'w')
    index = 0
    for photo in photos:
        recogniser = Recogniser(photo, index)
        recogniser.count(recogniser.findContours())
        index += 1
else:
    print("Вы используете ContourRecognition как библиотеку!")
