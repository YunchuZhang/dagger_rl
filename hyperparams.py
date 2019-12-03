import os
import getpass

# Updated the values 
B = 8 #4 # batch size
H = 64 #240 # height
W = 64 #320 # width
S = 4
####################

BY = 200*2 # bird height (y axis, [-40, 40])
BX = 176*2 # bird width (x axis, [0, 70.4])
BZ = 20 # bird depth (z axis, [-3.0, 1.0])

MH = 200*2
MW = 176*2
MD = 20

PH = int(128/4)
PW = int(384/4)

ZY = 32 
ZX = 32 
ZZ = 16 

N = 50 # number of boxes produced by the rcnn (not all are good)
K = 1 # number of boxes to actually use
# S = 2 # seq length
T = 256 # height & width of birdview map
V = 100000 # num velodyne points
