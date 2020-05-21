import cv2
import imutils
import os

def crop_detected_face(img,face):
    (x, y, w, h) = face
    crop_img = img[y:y+h, x:x+w]
    if crop_img is not None and  crop_img.size != 0:
        #crop_img = imutils.resize(crop_img, width=400)
        crop_img = cv2.resize(crop_img, (400,400), interpolation=cv2.INTER_AREA)
    return crop_img

def crop_face(path,err_path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    if len(faces_detected) != 1:
        for face in faces_detected:
            img = crop_detected_face(img,face)
            if img.size != 0:
                (name,num) = path.split('/')[-2:]
                err_path = err_path + 'err-' +name + name + '/' + num
                cv2.imwrite(err_path,img)
        return None
    else:
        return crop_detected_face(img,faces_detected[0])

path = '/home/snir/Projects/Python/hefers/'
raw_data_path = path + 'photos/raw-data/'
proccesed_path = path + 'photos/filtered/'
err_path = path + 'photos/errors/'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

names = os.listdir(raw_data_path)
for name in names:
    dst = proccesed_path + name + '/'
    imgs = os.listdir(raw_data_path + name)
    for img in imgs:
        img_path = raw_data_path + name + '/' + img
        face = crop_face(img_path,err_path)
        if(face is not None):
            cv2.imwrite(dst+img,face)
