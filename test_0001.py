import cv2

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cropped():
    cropped = image[30:500, 100:2000]  # image[y:y + высота, x:x + ширина].
    viewImage(cropped, "Dog after cropping")

def change_size():
    img = cv2.imread("dog2.jpeg")
    scale_percent = 10  # Процент от изначального размера
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTERSECT_FULL)
    viewImage(resized, "After changing on 20 %")

def rotate():
    image3 = cv2.imread("dog.jpeg")
    (h, w, d) = image3.shape
    center = (w // 2, h //2 - 100 )
    M = cv2.getRotationMatrix2D(center, 45, 4.5)
    rotated = cv2.warpAffine(image3, M, (w, h))
    viewImage(rotated, "Dog after rotation on 180")

def change_color():
    im = cv2.imread("dog.jpeg")
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
    ret, thresh1 = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(im, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(im, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(im, 127, 255, cv2.THRESH_TOZERO_INV)
    #ret, threshold_image = cv2.threshold(im, 150, 200, 10)
    viewImage(gray_image, "Dog in  the gray color")
    viewImage(threshold_image, "Black-white dog")
    viewImage(thresh1, "THRESH_BINARY")
    viewImage(thresh2, "THRESH_BINARY_INV")
    viewImage(thresh3, "THRESH_TRUNC")
    viewImage(thresh4, "THRESH_TOZERO")
    viewImage(thresh5, "THRESH_TOZERO_INV")

def blur():
    blurred = cv2.GaussianBlur(image, (111, 1111), 0)
    cv2.imwrite("test.png", blurred)
    viewImage(blurred, "Blured dog")

def rectangle():
    """
    Само изображение.
    Координата верхнего левого угла (x1, y1).
    Координата нижнего правого угла (x2, y2).
    Цвет прямоугольника (GBR/RGB в зависимости от выбранной цветовой модели).
    Толщина линии прямоугольника.
    :return:
    """
    output = image.copy()
    cv2.rectangle(output, (100, 400), (1000, 2400), (0, 255, 255), 100)
    viewImage(output, "Draw rectangle")
def line():
    """
    Само изображение, на котором рисуется линия.
    Координата первой точки (x1, y1).
    Координата второй точки (x2, y2).
    Цвет линии (GBR/RGB в зависимости от выбранной цветовой модели).
    Толщина линии.
    :return:
    """
    output = image.copy()
    cv2.line(output, (60, 20), (400, 200), (255, 0, 255), 5)
    viewImage(output, "2 dogs, divided by line")

def text():
    """
    Непосредственно изображение.
    Текст для изображения.
    Координата нижнего левого угла начала текста (x, y).
    Используемый шрифт.
    Размер шрифта.
    Цвет текста (GBR/RGB в зависимости от выбранной цветовой модели).
    Толщина линий букв.
    :return:
    """
    output = image.copy()
    cv2.putText(output, "We <3 Dogs", (1500, 3600), cv2.FONT_HERSHEY_SIMPLEX, 15, (30, 105, 210), 40)
    viewImage(output, "Image with a text label")


def recognize_image():
    """
    detectMultiScale — общая функция для распознавания как лиц, так и объектов. Чтобы функция искала именно лица, мы передаём ей соответствующий каскад.

    Функция detectMultiScale принимает 4 параметра:

    Обрабатываемое изображение в градации серого.
    Параметр scaleFactor. Некоторые лица могут быть больше других, поскольку находятся ближе, чем остальные. Этот параметр компенсирует перспективу.
    Алгоритм распознавания использует скользящее окно во время распознавания объектов. Параметр minNeighbors определяет количество объектов вокруг лица.
    То есть чем больше значение этого параметра, тем больше аналогичных объектов необходимо алгоритму, чтобы он определил текущий объект, как лицо.
    Слишком маленькое значение увеличит количество ложных срабатываний, а слишком большое сделает алгоритм более требовательным.
    minSize — непосредственно размер этих областей.
    :return:
    """
    image_path = "girls.jpeg"
    face_cascade = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10)
    )
    faces_detected = "Face detected: " + format(len(faces))
    print(faces_detected)
    # Рисуем квадраты вокруг лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    viewImage(image, faces_detected)

def save_img():
    image = cv2.imread("./импорт/путь.расширение")
    cv2.imwrite("./экспорт/путь.расширение", image)

image = cv2.imread("dog.jpeg")
cv2.imshow("Image", image)
#rotate()

cv2.waitKey(0)
#cropped()
#change_color()
#blur()
#change_color()
rectangle()
#line()
#recognize_image()
cv2.destroyAllWindows()