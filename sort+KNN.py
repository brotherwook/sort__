#%%

import pickle
import numpy as np
import os
# ### 현재 실행되는 python 파일의 위치 입력
# os.chdir( "D:/sort__" )

import sort_lib.SORT as SORT
import cv2
import time


#%% 영상 전처리
def imagepreprocessing(img):
    global dets
    global reshapeROI

    # grayScale 이미지로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # backgroundSubtracktor에 이미지를 넣어서 배경과 전경을 분리한 이미지 생성
    fgmask = fgbg.apply(gray)

    cv2.imshow("tew", fgmask)
    # 그림자제거
    ret, thresh = cv2.threshold(fgmask, 126, 255, cv2.THRESH_BINARY)

    # 노이즈 제거
    medianblur = cv2.medianBlur(thresh, 5)

    # 팽창연산
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    dil = cv2.dilate(medianblur, k)

    # 윤곽선찾기
    contours, hierachy = cv2.findContours(dil, mode, method)

    # 윤곽선 안을 흰색으로 채우기
    for i, contour in enumerate(contours):
        dil = cv2.fillPoly(dil, contour, 255)

    # 다시한번 팽창 연산
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    dil = cv2.dilate(dil, k)

    # close 연산 (팽창 후 침식)
    kernel = np.ones((5, 5), np.uint8)
    dil = cv2.morphologyEx(dil.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # 다시 윤곽선 찾기
    contours, hierachy = cv2.findContours(dil, mode, method)

    # 중심점 구하기
    for i, contour in enumerate(contours):
        dil = cv2.fillPoly(dil, contour, 255)
        x, y, w, h = cv2.boundingRect(contour)
        if x > reshapeROI[0] and y > reshapeROI[1] and x+w < reshapeROI[4] and y+h < reshapeROI[5]:
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
for_car = "C:/Users/user/Desktop/영상/인코딩/F20003_4_202011020900.avi"

# video 정보
cap = cv2.VideoCapture(for_car)
print(cap.get( cv2.CAP_PROP_FRAME_HEIGHT ))
_ , frame = cap.read()
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 또는 cap.get(3)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
fps = cap.get(cv2.CAP_PROP_FPS) # 또는 cap.get(5)

# save 객체
fourcc = cv2.VideoWriter_fourcc(*'H264') # 코덱 정의
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

    frame = cv2.resize(frame,(int(width/3),int(height/3)))

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

            cv2.imshow('yolo', frame)
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
 # ram할당 제거