import cv2
import numpy as np
import os
# ### 현재 실행되는 python 파일의 위치 입력
# os.chdir( "D:/sort__" )
# os.getcwd()

import sort_lib.SORT as SORT

# video_path = 'D:/video/20201008/F20003_4_202010080900_ROI.avi'
video_path = "./input/F20003_4_202011020900.avi"
img_name = video_path[-25:-4]

# 비디오 불러오기
cap = cv2.VideoCapture(video_path)
if (not cap.isOpened()):
    print('Error opening video')

height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
fps = cap.get(cv2.CAP_PROP_FPS) # 또는 cap.get(5)

acc_gray = np.zeros(shape=(int(height/2), int(width/2)), dtype=np.float32)
acc_bgr = np.zeros(shape=(int(height/2), int(width/2), 3), dtype=np.float32)
t = 0

mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
dets = []

# 동영상 저장용 (안씀)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 디지털 미디어 포맷 코드 생성 , 인코딩 방식 설
out = cv2.VideoWriter('./res.avi', fourcc, 30.0, (int(width/2), int(height/2)))


"""
tracker 객체의 parameter 정의 구간입니다.
"""
max_age =  20
min_hits =  10
iou_threshold = 0.3

tracker_test = SORT.Sort( max_age = max_age,
                     min_hits = min_hits,
                     iou_threshold = iou_threshold )

while True:
    try:
        retval, frame = cap.read()
        if not retval:
            break
        frame = cv2.resize(frame, (int(width/2), int(height/2)))
        t += 1

        cv2.accumulate(frame, acc_bgr)
        avg_bgr = acc_bgr / t
        dst_bgr = cv2.convertScaleAbs(avg_bgr)

        cv2.imshow("acc_bgr", dst_bgr)

        diff_bgr = cv2.absdiff(frame, dst_bgr)

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

        dets = []
        # 중심점 구하기
        for i, contour in enumerate(contours):
            dil = cv2.fillPoly(dil, contour, 255)
            x, y, w, h = cv2.boundingRect(contour)
            if 20 < w < 350 and 20 < h < 200:
                dets.append(np.array([y, x, y + h, x + w]))
        # cv2.imshow("dil", dil)

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
            p2 = d[3], d[2]
            cv2.rectangle(frame, p1, p2, (14, 255, 0), 1)
            cv2.putText(frame, str(d[4]), p1, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))

        # for det in dets:
        #     frame = cv2.rectangle(frame, (det[1], det[0]), (det[3],det[2]), (255,0,0), 2, cv2.LINE_AA)

        out.write(frame)
        # cv2.imshow("frame", frame)
        print("\r",t/(fps*360)*100,"%",end="")
        if cv2.waitKey(1) == 27:
            break

    except KeyboardInterrupt:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

