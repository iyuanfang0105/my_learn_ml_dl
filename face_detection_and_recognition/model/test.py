# coding=utf-8
import numpy as np
import cv2, os, time


def show_detections(image, detections):
    h, w, c = image.shape
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 1)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    return image


def detect_img(net, image):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    # start = time.time()
    detections = net.forward()
    # end = time.time()
    # print(end - start)
    return show_detections(image, detections)


def test_dir(net, dir="images"):
    files = os.listdir(dir)
    for file in files:
        filepath = dir + "/" + file
        img = cv2.imread(filepath)
        showimg = detect_img(net, img)
        cv2.imshow("img", showimg)
        cv2.waitKey()


def test_camera(net):
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        showimg = detect_img(net, img)
        cv2.imshow("img", showimg)
        cv2.waitKey(1)


if __name__ == "__main__":
    # net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    net =cv2.dnn.readNetFromTensorflow("Opencv-DNN-Shot-Multibox-Detector/opencv_face_detector_uint8.pb","Opencv-DNN-Shot-Multibox-Detector/opencv_face_detector.pbtxt")
    # test_dir(net)
    test_camera(net)
