import cv2
import pykitti
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

basedir = './'

date = '2011_09_30'
drive = '0018'
dataset = pykitti.raw(basedir, date, drive, imformat='cv2')

def cartesian(longitude, latitude, elevation):
    R = 6378137.0 + elevation  # relative to centre of the earth
    X = R * math.cos(longitude * math.pi / 180) * math.sin(latitude * math.pi / 180)
    Y = R * math.sin(longitude * math.pi / 180) * math.sin(latitude * math.pi / 180)
    Z = R * math.cos(latitude * math.pi / 180)
    return X,Y,Z

nframes = len(dataset.cam0_files)
# nframes = 20

fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot()

lat1 = dataset.oxts[0][0][0]
long1 = dataset.oxts[0][0][1]
ele1 = dataset.oxts[0][0][2]

x1,y1,z1 = cartesian(long1,lat1,ele1)
lt = []
lg = []

for i in tqdm(range(nframes-1)):
       lati = dataset.oxts[i][0][0]
       longi = dataset.oxts[i][0][1]
       elei = dataset.oxts[i][0][2]

       xi,yi,zi = cartesian(longi,lati,elei)
       lt.append(xi-x1)
       lg.append(yi-y1)

plt.plot(lt,lg)
plt.show()