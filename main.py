import cv2
import pykitti
import numpy as np
import matplotlib.pyplot as plt

basedir = './'

date = '2011_09_30'
drive = '0018'


dataset = pykitti.raw(basedir, date, drive)
first_gray = dataset.get_gray(0)

img_np = np.array(first_gray[0])
img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


cv2.namedWindow("Input")
cv2.imshow("Input",img_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()