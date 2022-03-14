from five_point_algorithm import *
import numpy as np

# numpyの行列をprintするときの設定
# 指数表示をしない
np.set_printoptions(suppress=True)

pts1 = np.array([
	[288.8398,  12.1534, 317.5059,  74.5754,  44.1327],
	[77.2382 , 163.7803,  82.8476, 220.5643, 192.9634]
], dtype=np.float64)

pts2 = np.array([
	[286.1892,   6.9846, 312.5673,  70.0283,  40.2131],
	[ 76.7289, 164.3921,  81.3119, 220.4840, 194.1166]
], dtype=np.float64)

K = np.array([
	[602.5277, 0       , 177.3328],
	[0       , 562.9129, 102.8893],
	[0       , 0       , 1.0000  ]
], dtype=np.float64)

E, R, t, Eo = five_point_algorithm(pts1, pts2, K, K)
pts3Dhomo = triangulate(pts1, pts2, K, K, E[0], R[0], t[0])
pts3D_1 = pts3Dhomo[0:4, :] / (np.ones((4, 1)) * pts3Dhomo[3, :])
pts3D_2 = np.dot(np.vstack((np.hstack((R[0], t[0].reshape((3, 1)))), np.array([0, 0, 0, 1], dtype=np.float64))), pts3D_1)
pts1_ = np.dot(K, pts3D_1[0:3, :] / (np.ones((3, 1)) * pts3D_1[2, :]))[0:2]
pts2_ = np.dot(K, pts3D_2[0:3, :] / (np.ones((3, 1)) * pts3D_2[2, :]))[0:2]

print(pts1_)
print(pts2_)
