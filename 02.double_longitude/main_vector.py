import cv2
import numpy as np
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

    # 数组， 循环每个点
    range_arr = np.array([range(R)])

    theta = Pi - (Pi/R)*(range_arr.T)
    temp_theta = np.tan(theta)**2

    phi = Pi - (Pi/R)*range_arr
    temp_phi = np.tan(phi)**2

    tempu = r/(temp_phi + 1 + temp_phi/temp_theta)**0.5
    tempv = r/(temp_theta + 1 + temp_theta/temp_phi)**0.5

    # 用于修正正负号
    flag = np.array([-1] * r + [1] * r)

    # 加0.5是为了四舍五入求最近点
    u = x0 + tempu * flag + 0.5
    v = y0 + tempv * np.array([flag]).T + 0.5

    # 防止数组溢出
    u[u<0]=0
    u[u>(src_w-1)] = src_w-1
    v[v<0]=0
    v[v>(src_h-1)] = src_h-1

    # 插值
    dst[:, :, :] = src[v.astype(int),u.astype(int)]
    return dst

if __name__ == "__main__":
    t = time.perf_counter()
    frame = cv2.imread('../imgs/pig.jpg')
    cut_img,R = cut(frame)
    t = time.perf_counter()
    result_img = undistort(cut_img,R)
    cv2.imwrite('../imgs/pig_vector_nearest.jpg',result_img)
    print(time.perf_counter()-t)