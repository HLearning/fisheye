import cv2
import numpy as np
import math
import time

# 鱼眼有效区域截取
def cut(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x,y,w,h = cv2.boundingRect(cnts)
    r = max(w/ 2, h/ 2)
    # 提取有效区域
    img_valid = img[y:y+h, x:x+w]
    return img_valid, int(r)

# 鱼眼矫正
def undistort(src,r):
    # r： 半径， R: 直径
    R = 2*r
    # Pi: 圆周率
    Pi = np.pi
    # 存储映射结果
    dst = np.zeros((R, R, 3))
    src_h, src_w, _ = src.shape
    # 圆心
    x0, y0 = src_w//2, src_h//2

    for dst_y in range(0, R):

        theta =  Pi - (Pi/R)*dst_y
        temp_theta = math.tan(theta)**2

        for dst_x in range(0, R):
            # 取坐标点 p[i][j]
            # 计算 sita 和 fi

            phi = Pi - (Pi/R)*dst_x
            temp_phi = math.tan(phi)**2

            tempu = r/(temp_phi+ 1 + temp_phi/temp_theta)**0.5
            tempv = r/(temp_theta + 1 + temp_theta/temp_phi)**0.5

            if (phi < Pi/2):
                u = x0 + tempu
            else:
                u = x0 - tempu

            if (theta < Pi/2):
                v = y0 + tempv
            else:
                v = y0 - tempv

            if (u>=0 and v>=0 and u+0.5<src_w and v+0.5<src_h):
                dst[dst_y, dst_x, :] = src[int(v+0.5)][int(u+0.5)]

                # 计算在源图上四个近邻点的位置
                # src_x, src_y = u, v
                # src_x_0 = int(src_x)
                # src_y_0 = int(src_y)
                # src_x_1 = min(src_x_0 + 1, src_w - 1)
                # src_y_1 = min(src_y_0 + 1, src_h - 1)
                #
                # value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0, :] + (src_x - src_x_0) * src[src_y_0, src_x_1, :]
                # value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0, :] + (src_x - src_x_0) * src[src_y_1, src_x_1, :]
                # dst[dst_y, dst_x, :] = ((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1 + 0.5).astype('uint8')

    return dst

if __name__ == "__main__":
    t = time.perf_counter()
    frame = cv2.imread('../imgs/pig.jpg')
    cut_img,R = cut(frame)
    result_img = undistort(cut_img,R)
    cv2.imwrite('../imgs/pig_nearest.jpg',result_img)
    print(time.perf_counter()-t)