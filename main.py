import cv2
import numpy as np
import sys
import os

print(1)

def getImagesAndLabels(path):
    # Создаем список файлов в папке patch
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    files = os.listdir(path)
    face=[] # Тут храним масив картинок
    ids = [] # Храним id лица
    #os.chdir(path)
    for image_path in imagePaths:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face.append(img)
        # Получаем id из названия
        id = int(os.path.split(image_path)[-1].split("_")[0])
        ids.append(id)
    return face,ids

def train(path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('face.yml')

def base(directory, name, face_id):
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    files = os.listdir(directory)
    count = 0

    current_path = os.getcwd()
    i_for_dir = 0
    path_name = 'base_faces_' + name  + str(i_for_dir)
    new_path = current_path + '\\' + path_name
    name_image = 'frame_'
    i = 0

    while True:
        try:
            os.mkdir(new_path)
            break
        except OSError as e:
            i_for_dir += 1
            path_name = 'base_faces_' + name  + str(i_for_dir)
            new_path = current_path + '\\' + path_name

    os.chdir(new_path)
    for image_path in files:
        image_path = directory+"\\"+image_path
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Сохраняем лицо
            count += 1
            cv2.imwrite(str(face_id) + '_' + str(count) + '.jpg', gray[y:y + h, x:x + w])
    cv2.destroyAllWindows()
    return new_path

def cut_video():
    path_to_video = input() #D:\Nauchka\Targer\11.mp4"
    current_path = os.getcwd()

    i_for_dir = 0
    path_name = 'cut_video_' + str(i_for_dir)
    new_path = current_path + '\\' + path_name
    name_image = 'frame_'
    i = 0

    while True:
        try:
            os.mkdir(new_path)
            break
        except OSError as e:
            i_for_dir += 1
            path_name = 'cut_video_' + str(i_for_dir)
            new_path = current_path + '\\' + path_name

    os.chdir(new_path)
    video = cv2.VideoCapture(path_to_video)
    FPS = video.get(cv2.CAP_PROP_FPS)
    skip = int(FPS)
    while True:
        # пропускаем некоторые кадры
        for u in range(skip):
            video.grab()
        ret, image = video.read()

        if image is None:
            break

        cv2.imwrite(name_image + str(i) + '.jpg', image)
        i += 1

    video.release()
    return new_path

def total(path_to_video):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('face.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    # Тип шрифта
    font = cv2.FONT_HERSHEY_SIMPLEX

    # iniciate id counter
    id = 0

    # Список имен для id
    names = ['None', 'Putin']

    cam = cv2.VideoCapture(path_to_video)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(10, 10),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Проверяем что лицо распознано
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # 'ESC' для Выхода
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

def test(path_to_video):
    cam = cv2.VideoCapture(path_to_video)
    while True:
        ret, img = cam.read()
        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # 'ESC' для Выхода
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()

def faces_data(path_to_video):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('face.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    # Тип шрифта
    font = cv2.FONT_HERSHEY_SIMPLEX

    # iniciate id counter
    id = 0

    # Список имен для id
    names = ['None', 'Putin']
    current_path = os.getcwd()
    i_for_dir = 0
    path_name = 'faces_data_' + str(i_for_dir)
    new_path = current_path + '\\' + path_name
    name_image = 'frame_'
    i = 0

    while True:
        try:
            os.mkdir(new_path)
            break
        except OSError as e:
            i_for_dir += 1
            path_name = 'faces_data_' + str(i_for_dir)
            new_path = current_path + '\\' + path_name

    os.chdir(new_path)
    video = cv2.VideoCapture(path_to_video)
    FPS = video.get(cv2.CAP_PROP_FPS)
    skip = int(FPS)

    while True:
        # пропускаем некоторые кадры
        for u in range(skip):
            video.grab()
        ret, img = video.read()

        if img is None:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(10, 10),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            temp = False
            # Проверяем что лицо распознано
            if (confidence < 100):
                temp = round(100 - confidence)>40
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
                if (temp):
                    cv2.imwrite(id + str(i) + '.jpg', img)
                    i+=1
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            # cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            # cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        #cv2.imshow('camera', img)


    video.release()
    return new_path

#path = cut_video()
#D:\Nauchka\Putin
#path = input("Directory: ")
#train(base("D:\\Nauchka\\Putin", "Putin", 1))
#train("D:\\Nauchka\\Putin1")
#faces_data("D:\\Nauchka\\Targer\\13.mp4")
total("D:\\Nauchka\\Targer\\13.mp4")
#test("D:\\Nauchka\\Targer\\12.mp4")
print(1)
