import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime, time
import time


def current_time():
    now = datetime.now().isoformat()
    return now


config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)
classLabels = []
label = 'cate.txt'
with open(label, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
# print(classLabels)

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

cap = cv2.VideoCapture('HSE.mp4')
cx = []
cy = []

while True:
    ret, img = cap.read()
    font_scale = 1
    font = cv2.FONT_HERSHEY_PLAIN

    # RED ZONE AREA
    area = [(181, 45), (306, 45), (306, 232), (181, 232)]
    cv2.polylines(img, [np.array(area, np.int32)], True, (0, 0, 255), 6)
    cv2.putText(img, "Red Zone", (275, 40), font,
                fontScale=font_scale,  color=(0, 0, 250), thickness=2)

    # DETECTION PERSON
    ClassIndex, confi, bbox = model.detect(img, confThreshold=0.5)
    #print("Indeks Kelas:", ClassIndex)
    for i in range(0, len(ClassIndex)):
        for classin, conf, boxes in zip(ClassIndex, confi, bbox):
            if classin == 1:  # PERSON CATEGORY
                #print(f"kelas ke-{i} berindex kelas: {classin}")
                box = bbox
                cx = int((box[i][0]+box[i][0]+box[i][2])/2)
                cy = int((box[i][1]+box[i][1]+box[i][3])/2)

                # Test Entering
                result = cv2.pointPolygonTest(
                    np.array(area, np.int32), (cx, cy), False)

                if result == 1:
                    print("[+] ALERT: RED ZONE OCCUPIED!!")
                else:
                    break

                print(
                    f"Time: {current_time()}:: {classLabels[classin-1]} is Detected in central coord (x,y): {cx,cy}")
                print(
                    "===================================================================================================\n")
                cv2.rectangle(img, bbox[i], (255, 0, 0), 2)
                cv2.putText(img, classLabels[classin-1], (bbox[i][0]+10, bbox[i][1]+40),
                            font, fontScale=font_scale, color=(0, 255, 0), thickness=1)
                cv2.circle(img, (cx, cy), radius=5,
                           color=(0, 0, 255), thickness=5)
            else:
                print("[-] No Person Detected")

    cv2.imshow("Img", img)

    if cv2.waitKey(30) == ord('q'):  # 30 fps video
        break

cap.release()
cv2.destroyAllWindows()
