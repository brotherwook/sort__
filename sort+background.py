#%%

import pickle
import numpy as np
import os
import sort_lib.SORT as SORT
import cv2
import time


#%% 영상 전처리
def imagepreprocessing(img):
    global dets
    global bkg_bgr

    diff_bgr = cv2.absdiff(img, bkg_bgr)

    db, dg, dr = cv2.split(diff_bgr)
    ret, bb = cv2.threshold(db, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, bg = cv2.threshold(dg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, br = cv2.threshold(dr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("bb", bb)
    # cv2.imshow("bg", bg)
    # cv2.imshow("br", br)

    bImage = cv2.bitwise_or(bb, bg)
    bImage = cv2.bitwise_or(br, bImage)

    median = cv2.medianBlur(bImage, 5)

    dil = cv2.dilate(median, None, 10)

    contours, hierarchy = cv2.findContours(dil, mode, method)

    # 중심점 구하기
    for i, contour in enumerate(contours):
        dil = cv2.fillPoly(dil, contour, 255)
        x, y, w, h = cv2.boundingRect(contour)
        if 20 < w < 350 and 20 < h < 200:
            dets.append(np.array([y,x,y+h,x+w]))
    cv2.imshow("dil",dil)



#%%
click = None
x1,y1 = None, None
roi = []
def MouseLeftClick(event, x, y, flags, param):
    global click
    global x1, y1
    global roi

    image = clone.copy()

    if event == cv2.EVENT_RBUTTONDOWN:
        # 마우스를 누른 상태
        roi = []
        click = True
        x1, y1 = x,y
        roi.append((x1,y1))
        # print("사각형의 왼쪽위 설정 : (" + str(x1) + ", " + str(y1) + ")")

    elif event == cv2.EVENT_MOUSEMOVE:
        # 마우스 이동
        # 마우스를 누른 상태 일경우
        if click == True:
            cv2.rectangle(image,(x1,y1),(x,y),(255,0,0), 2)
            # print("(" + str(x1) + ", " + str(y1) + "), (" + str(x) + ", " + str(y) + ")")
            cv2.imshow("yolo", image)

    elif event == cv2.EVENT_RBUTTONUP:
        # 마우스를 때면 상태 변경
        click = False
        cv2.rectangle(image,(x1,y1),(x,y),(255,0,0), 2)
        cv2.imshow("yolo", image)
        roi.append((x1, y))
        roi.append((x, y))
        roi.append((x, y1))
        print("width=",x-x1,"height=",y-y1)

    elif event == cv2.EVENT_LBUTTONDBLCLK:
        print("X:",x,"Y:",y)



# 새 윈도우 창을 만들고 그 윈도우 창에 MouseLeftClick 함수를 세팅해 줍니다.
cv2.namedWindow("yolo")
cv2.setMouseCallback("yolo", MouseLeftClick)

#%%

"""
tracker 객체의 parameter 정의 구간입니다.
"""
max_age =  20
min_hits =  10
iou_threshold = 0.3

tracker_test = SORT.Sort( max_age = max_age,
                     min_hits = min_hits,
                     iou_threshold = iou_threshold )


# load_video
video_path = "./input/F20003_4_202011020900.avi"
img_name = video_path[-25:-4]

background_path = "D:/sort__/background/" + img_name + ".png"

bkg_bgr = cv2.imread(background_path)

# video 정보
cap = cv2.VideoCapture(video_path)
print(cap.get( cv2.CAP_PROP_FRAME_HEIGHT ))
_ , frame = cap.read()
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 또는 cap.get(3)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
fps = cap.get(cv2.CAP_PROP_FPS) # 또는 cap.get(5)

# save 객체
fourcc = cv2.VideoWriter_fourcc(*'DIVX') # 코덱 정의
out = cv2.VideoWriter('res.avi', fourcc, fps, (int(width), int(height)))

"""
data shape 정의
매 프레임 마다 객체 식별후 tracker 에게
"""

fgbg = cv2.createBackgroundSubtractorKNN()
mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

dets = []

start_time = time.time()
cnt = 0

flag = 0
temp = []
reshapeROI = None

while cap.isOpened():
    ret , frame = cap.read()
    cnt += 1
    if not ret:
        break

    frame = cv2.resize(frame,(int(width/2),int(height/2)))
    bkg_bgr = cv2.resize(bkg_bgr,(int(width/2),int(height/2)))
    """
    프레임당 객체를 식별 하고
    """
    if cnt % 6 != 0:

        if flag == 0:
            flag = 1
            clone = frame.copy()
            while True:
                if len(roi) > 1:
                    cv2.rectangle(frame, roi[0], roi[2], color=(0, 0, 255), thickness=1)
                    mask = np.zeros_like(frame)
                    roi = np.array(roi)
                    roi = roi[np.newaxis, ...]
                    cv2.fillPoly(mask, roi, (255, 255, 255))
                    reshapeROI = roi.reshape(-1)
                    temp.append(roi[0])

                cv2.imshow("yolo", frame)
                key = cv2.waitKey(0)

                if key == ord("r"):
                    flag = 1
                    break

        if flag == 1:
            if reshapeROI is None:
                reshapeROI = [0, 0, 0, height, width, height, width, 0]
            dets = []
            imagepreprocessing(frame)

            """
            프레임의 탐지된 객체들을 tracker에 부여 및 tracker 갱신합니다
            $$dets$$
            shape : ( 해당프레임 탐지된 객체수 , 4 )
            구조 : [ 
                    [ x1, y1, x2, y2 ],
                    [ x1, y1, x2, y2 ],..
                ]
            """
            try:
                trackers = tracker_test.update(np.array(dets))
            except:
                pass

            """
            프레임별 갱신된 정보를 출력합니다
            """
            for d in trackers:
                # print(frame, d[4], d[:4])
                d = d.astype(np.int32)
                p1 = d[1], d[0]
                p2 = d[3] , d[2]
                cv2.rectangle( frame , p1 , p2 , ( 14 , 255 , 0 ) , 1 )
                cv2.putText( frame , str(d[4]) , p1 , cv2.FONT_HERSHEY_DUPLEX , 1, (0,0,0) )

            for j in range(len(temp)):
                for i, v in enumerate(roi[0]):
                    if i < len(roi[0]) - 1:
                        frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][i + 1]), (0, 0, 255), 2)
                    else:
                        frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][0]), (0, 0, 255), 2)

            cv2.imshow('yolo' , frame )
            out.write(frame)

            # time.sleep(10)

            if cv2.waitKey(1) == 27:
                flag = 2
                break

        else:
            break

print( f"총 소요 시간 : {(time.time() - start_time)/60} 분")

out.release()
cap.release()
cv2.destroyAllWindows()
