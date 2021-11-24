import cv2
import pykitti
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

class GPS:
    def __init__(self,datset):
        self.dataset = dataset
        self.initial = self.cartesian(0)


    def cartesian(self, frame):
        oxts_packet = self.dataset.oxts[frame].packet
        elevation = oxts_packet.alt
        longitude = oxts_packet.lon
        latitude = oxts_packet.lat

        R = 6378137.0 + elevation  # relative to centre of the earth
        X = R * math.cos(longitude * math.pi / 180) * math.sin(latitude * math.pi / 180)
        Y = R * math.sin(longitude * math.pi / 180) * math.sin(latitude * math.pi / 180)

        if(frame != 0):
            y = Y - self.initial[1]
            x = X - self.initial[0]
        else:
            x = X
            y = Y

        return (x,y)



basedir = './'
date = '2011_09_30'
drive = '0018'


dataset = pykitti.raw(basedir, date, drive, imformat='cv2')

# dataset.calib:         Calibration data are accessible as a named tuple
# dataset.timestamps:    Timestamps are parsed into a list of datetime objects
# dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
# dataset.camN:          Returns a generator that loads individual images from camera N
# dataset.get_camN(idx): Returns the image from camera N at idx  
# dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
# dataset.get_gray(idx): Returns the monochrome stereo pair at idx  
# dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
# dataset.get_rgb(idx):  Returns the RGB stereo pair at idx  
# dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
# dataset.get_velo(idx): Returns the velodyne scan at idx  

gps = GPS(dataset)
n_frames = len(dataset.cam0_files)

gps_data = []
imu_data = []

for frame in tqdm(range(1,n_frames-1)):
    gps_data.append(gps.cartesian(frame))


    






plt.plot(*zip(*gps_data))
plt.show()
plt.close()




# first_gray = dataset.get_gray(0)

# print(dataset.oxts[0].packet)

# img_np = np.array(first_gray[0])
# img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

# debug = Debugger((2,1))

# debug.collect(img_cv2,("image","gray"))
# debug.collect(first_gray[1],("image","gray"))
# debug.display(plot=True)


# cv2.namedWindow("Input")
# cv2.imshow("Input",img_cv2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()