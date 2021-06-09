import cv2
import numpy as np
def lpr(filename):
  img = cv2.imread(filename)
  img = cv2.resize(img,(1000,500))
  '''
  依序將照片灰化， gaussian 處理， sobel 處理，將照片二值化
   gaussian 函數的第四個引數設置為零，表示不計算 y 方向的梯度，因為車牌數字在豎方向較長，重點在於得到豎方向的 edge
  '''
  gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  GaussianBlur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
  Sobel_img = cv2.Sobel(GaussianBlur_img, -1, 1, 0, ksize=3)
  ret, binary_img = cv2.threshold(Sobel_img, 127, 255, cv2.THRESH_BINARY)
  '''
  型態學運算
  '''
  kernel = np.ones((5, 15), np.uint8)
  '''
  先閉運算，將車牌數字連接，再開運算，將不是塊狀的，或是較小的部分刪掉
  '''
  close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
  open_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel)
  #kernel2 = np.ones((10, 10), np.uint8)
  #open_img2 = cv2.morphologyEx(open_img, cv2.MORPH_OPEN, kernel2)
  '''
  由於得到的輪廓 edge 不整齊，因此再做一次膨脹處理
  '''
  element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
  dilation_img = cv2.dilate(open_img, element, iterations=3)
  '''
  獲取輪廓
  '''
  contours, hierarchy = cv2.findContours(dilation_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  #cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
  #cv2.imshow("lpr", img)
  #cv2.waitKey(0)
  '''
  將輪廓轉換為長方型
  '''
  rectangles = []
  for c in contours:
    x = []
    y = []
    for point in c:
      y.append(point[0][0])
      x.append(point[0][1])
    r = [min(y), min(x), max(y), max(x)]
    rectangles.append(r)
  '''
  資料說：將 rgb 彩色的車牌照片轉換至 hsv 空間，在 hsv 空間裡，車牌是藍色的，所以用藍色識別出車牌區域
  '''
  dist_r = []
  max_mean = 0
  for r in rectangles:
    block = img[r[1]:r[3], r[0]:r[2]]
    hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
    low = np.array([100, 60, 60])
    up = np.array([140, 255, 255])
    result = cv2.inRange(hsv, low, up)
    '''
    用計算 mean 的方式找藍色最多的區塊
    '''
    mean = cv2.mean(result)
    if mean[0] > max_mean:
      max_mean = mean[0]
      dist_r = r
  cv2.rectangle(img, (dist_r[0], dist_r[1]), (dist_r[2], dist_r[3]), (0, 255, 0), 2)
  cv2.imshow("lpr", img)
  cv2.waitKey(0)
lpr('10.jpeg')
