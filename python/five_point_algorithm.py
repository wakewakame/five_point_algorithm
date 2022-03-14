import numpy as np
import scipy.linalg

# function [E_all, R_all, t_all, Eo_all] = five_point_algorithm( pts1, pts2, K1, K2 )
def five_point_algorithm(pts1, pts2, K1, K2):
	# 引数に渡された行列のサイズチェック
	if pts1.shape != (2, 5) or pts2.shape != (2, 5):
		print("warning: pts1 and pts2 must be of size 2x5")
		return None
	if K1.shape != (3, 3) or K2.shape != (3, 3):
		print("warning: K1 and K2 must be of size 3x3")
		return None

	# 引数に渡された画像座標系の5つの点を正規化画像座標系に変換
	N = 5
	K1_inv = np.array([[1.0/K1[0,0],0.0,-K1[0,2]/K1[0,0]],[0.0,1.0/K1[1,1],-K1[1,2]/K1[1,1]],[0,0,1]], dtype=np.float64)
	K2_inv = np.array([[1.0/K2[0,0],0.0,-K2[0,2]/K2[0,0]],[0.0,1.0/K2[1,1],-K2[1,2]/K2[1,1]],[0,0,1]], dtype=np.float64)
	q1 = np.dot(K1_inv, np.vstack((pts1, np.ones((1, N), dtype=np.float64))))
	q2 = np.dot(K2_inv, np.vstack((pts2, np.ones((1, N), dtype=np.float64))))

	q = np.array([
		q1[0,:] * q2[0,:], q1[1,:] * q2[0,:], q1[2,:] * q2[0,:],
		q1[0,:] * q2[1,:], q1[1,:] * q2[1,:], q1[2,:] * q2[1,:],
		q1[0,:] * q2[2,:], q1[1,:] * q2[2,:], q1[2,:] * q2[2,:]
	]).transpose()

	# SVDから零空間の計算
	# 参考元: http://roboticstips.web.fc2.com/index/mathematics/svd/svd.html
	nullSpace = np.linalg.svd(q)[2].transpose()[:,5:]
	X = nullSpace[:,0]
	Y = nullSpace[:,1]
	Z = nullSpace[:,2]
	W = nullSpace[:,3]

	Xmat = X.reshape((3, 3))
	Ymat = Y.reshape((3, 3))
	Zmat = Z.reshape((3, 3))
	Wmat = W.reshape((3, 3))

	X_ = np.dot(np.dot(K2_inv.transpose(), Xmat), K1_inv)
	Y_ = np.dot(np.dot(K2_inv.transpose(), Ymat), K1_inv)
	Z_ = np.dot(np.dot(K2_inv.transpose(), Zmat), K1_inv)
	W_ = np.dot(np.dot(K2_inv.transpose(), Wmat), K1_inv)

	detF = (p2p1(p1p1([X_[0,1],Y_[0,1],Z_[0,1],W_[0,1]],
	                  [X_[1,2],Y_[1,2],Z_[1,2],W_[1,2]]) -
	             p1p1([X_[0,2],Y_[0,2],Z_[0,2],W_[0,2]],
	                  [X_[1,1],Y_[1,1],Z_[1,1],W_[1,1]]),
	             [X_[2,0],Y_[2,0],Z_[2,0],W_[2,0]]) +
	        p2p1(p1p1([X_[0,2],Y_[0,2],Z_[0,2],W_[0,2]],
	                  [X_[1,0],Y_[1,0],Z_[1,0],W_[1,0]]) -
	             p1p1([X_[0,0],Y_[0,0],Z_[0,0],W_[0,0]],
	                  [X_[1,2],Y_[1,2],Z_[1,2],W_[1,2]]),
	             [X_[2,1],Y_[2,1],Z_[2,1],W_[2,1]]) +
	        p2p1(p1p1([X_[0,0],Y_[0,0],Z_[0,0],W_[0,0]],
	                  [X_[1,1],Y_[1,1],Z_[1,1],W_[1,1]]) -
	             p1p1([X_[0,1],Y_[0,1],Z_[0,1],W_[0,1]],
	                  [X_[1,0],Y_[1,0],Z_[1,0],W_[1,0]]),
	             [X_[2,2],Y_[2,2],Z_[2,2],W_[2,2]]))

	EE_t11 = (p1p1([Xmat[0,0],Ymat[0,0],Zmat[0,0],Wmat[0,0]],
				   [Xmat[0,0],Ymat[0,0],Zmat[0,0],Wmat[0,0]]) +
			  p1p1([Xmat[0,1],Ymat[0,1],Zmat[0,1],Wmat[0,1]],
				   [Xmat[0,1],Ymat[0,1],Zmat[0,1],Wmat[0,1]]) +
			  p1p1([Xmat[0,2],Ymat[0,2],Zmat[0,2],Wmat[0,2]],
				   [Xmat[0,2],Ymat[0,2],Zmat[0,2],Wmat[0,2]]))
	EE_t12 = (p1p1([Xmat[0,0],Ymat[0,0],Zmat[0,0],Wmat[0,0]],
				   [Xmat[1,0],Ymat[1,0],Zmat[1,0],Wmat[1,0]]) +
			  p1p1([Xmat[0,1],Ymat[0,1],Zmat[0,1],Wmat[0,1]],
				   [Xmat[1,1],Ymat[1,1],Zmat[1,1],Wmat[1,1]]) +
			  p1p1([Xmat[0,2],Ymat[0,2],Zmat[0,2],Wmat[0,2]],
				   [Xmat[1,2],Ymat[1,2],Zmat[1,2],Wmat[1,2]]))
	EE_t13 = (p1p1([Xmat[0,0],Ymat[0,0],Zmat[0,0],Wmat[0,0]],
				   [Xmat[2,0],Ymat[2,0],Zmat[2,0],Wmat[2,0]]) +
			  p1p1([Xmat[0,1],Ymat[0,1],Zmat[0,1],Wmat[0,1]],
				   [Xmat[2,1],Ymat[2,1],Zmat[2,1],Wmat[2,1]]) +
			  p1p1([Xmat[0,2],Ymat[0,2],Zmat[0,2],Wmat[0,2]],
				   [Xmat[2,2],Ymat[2,2],Zmat[2,2],Wmat[2,2]]))
	EE_t22 = (p1p1([Xmat[1,0],Ymat[1,0],Zmat[1,0],Wmat[1,0]],
				   [Xmat[1,0],Ymat[1,0],Zmat[1,0],Wmat[1,0]]) +
			  p1p1([Xmat[1,1],Ymat[1,1],Zmat[1,1],Wmat[1,1]],
				   [Xmat[1,1],Ymat[1,1],Zmat[1,1],Wmat[1,1]]) +
			  p1p1([Xmat[1,2],Ymat[1,2],Zmat[1,2],Wmat[1,2]],
				   [Xmat[1,2],Ymat[1,2],Zmat[1,2],Wmat[1,2]]))
	EE_t23 = (p1p1([Xmat[1,0],Ymat[1,0],Zmat[1,0],Wmat[1,0]],
				   [Xmat[2,0],Ymat[2,0],Zmat[2,0],Wmat[2,0]]) +
			  p1p1([Xmat[1,1],Ymat[1,1],Zmat[1,1],Wmat[1,1]],
				   [Xmat[2,1],Ymat[2,1],Zmat[2,1],Wmat[2,1]]) +
			  p1p1([Xmat[1,2],Ymat[1,2],Zmat[1,2],Wmat[1,2]],
				   [Xmat[2,2],Ymat[2,2],Zmat[2,2],Wmat[2,2]]))
	EE_t33 = (p1p1([Xmat[2,0],Ymat[2,0],Zmat[2,0],Wmat[2,0]],
				   [Xmat[2,0],Ymat[2,0],Zmat[2,0],Wmat[2,0]]) +
			  p1p1([Xmat[2,1],Ymat[2,1],Zmat[2,1],Wmat[2,1]],
				   [Xmat[2,1],Ymat[2,1],Zmat[2,1],Wmat[2,1]]) +
			  p1p1([Xmat[2,2],Ymat[2,2],Zmat[2,2],Wmat[2,2]],
				   [Xmat[2,2],Ymat[2,2],Zmat[2,2],Wmat[2,2]]))

	A_11 = EE_t11 - 0.5 * (EE_t11 + EE_t22 + EE_t33)
	A_12 = EE_t12
	A_13 = EE_t13
	A_21 = A_12
	A_22 = EE_t22 - 0.5 * (EE_t11 + EE_t22 + EE_t33)
	A_23 = EE_t23
	A_31 = A_13
	A_32 = A_23
	A_33 = EE_t33 - 0.5 * (EE_t11 + EE_t22 + EE_t33)

	AE_11 = (p2p1(A_11, [Xmat[0,0],Ymat[0,0],Zmat[0,0],Wmat[0,0]]) +
			 p2p1(A_12, [Xmat[1,0],Ymat[1,0],Zmat[1,0],Wmat[1,0]]) +
			 p2p1(A_13, [Xmat[2,0],Ymat[2,0],Zmat[2,0],Wmat[2,0]]))
	AE_12 = (p2p1(A_11, [Xmat[0,1],Ymat[0,1],Zmat[0,1],Wmat[0,1]]) +
			 p2p1(A_12, [Xmat[1,1],Ymat[1,1],Zmat[1,1],Wmat[1,1]]) +
			 p2p1(A_13, [Xmat[2,1],Ymat[2,1],Zmat[2,1],Wmat[2,1]]))
	AE_13 = (p2p1(A_11, [Xmat[0,2],Ymat[0,2],Zmat[0,2],Wmat[0,2]]) +
			 p2p1(A_12, [Xmat[1,2],Ymat[1,2],Zmat[1,2],Wmat[1,2]]) +
			 p2p1(A_13, [Xmat[2,2],Ymat[2,2],Zmat[2,2],Wmat[2,2]]))
	AE_21 = (p2p1(A_21, [Xmat[0,0],Ymat[0,0],Zmat[0,0],Wmat[0,0]]) +
			 p2p1(A_22, [Xmat[1,0],Ymat[1,0],Zmat[1,0],Wmat[1,0]]) +
			 p2p1(A_23, [Xmat[2,0],Ymat[2,0],Zmat[2,0],Wmat[2,0]]))
	AE_22 = (p2p1(A_21, [Xmat[0,1],Ymat[0,1],Zmat[0,1],Wmat[0,1]]) +
			 p2p1(A_22, [Xmat[1,1],Ymat[1,1],Zmat[1,1],Wmat[1,1]]) +
			 p2p1(A_23, [Xmat[2,1],Ymat[2,1],Zmat[2,1],Wmat[2,1]]))
	AE_23 = (p2p1(A_21, [Xmat[0,2],Ymat[0,2],Zmat[0,2],Wmat[0,2]]) +
			 p2p1(A_22, [Xmat[1,2],Ymat[1,2],Zmat[1,2],Wmat[1,2]]) +
			 p2p1(A_23, [Xmat[2,2],Ymat[2,2],Zmat[2,2],Wmat[2,2]]))
	AE_31 = (p2p1(A_31, [Xmat[0,0],Ymat[0,0],Zmat[0,0],Wmat[0,0]]) +
			 p2p1(A_32, [Xmat[1,0],Ymat[1,0],Zmat[1,0],Wmat[1,0]]) +
			 p2p1(A_33, [Xmat[2,0],Ymat[2,0],Zmat[2,0],Wmat[2,0]]))
	AE_32 = (p2p1(A_31, [Xmat[0,1],Ymat[0,1],Zmat[0,1],Wmat[0,1]]) +
			 p2p1(A_32, [Xmat[1,1],Ymat[1,1],Zmat[1,1],Wmat[1,1]]) +
			 p2p1(A_33, [Xmat[2,1],Ymat[2,1],Zmat[2,1],Wmat[2,1]]))
	AE_33 = (p2p1(A_31, [Xmat[0,2],Ymat[0,2],Zmat[0,2],Wmat[0,2]]) +
			 p2p1(A_32, [Xmat[1,2],Ymat[1,2],Zmat[1,2],Wmat[1,2]]) +
			 p2p1(A_33, [Xmat[2,2],Ymat[2,2],Zmat[2,2],Wmat[2,2]]))

	A = np.array([detF, AE_11, AE_12, AE_13, AE_21, AE_22, AE_23, AE_31, AE_32, AE_33])
	A = A[:, [0,1,3,4,5,10,7,11,9,13,6,14,16,8,15,17,2,12,18,19]]

	A_el = gj_elim_pp(A)

	k_row = partial_subtrc(A_el[4,10:20], A_el[5,10:20])
	l_row = partial_subtrc(A_el[6,10:20], A_el[7,10:20])
	m_row = partial_subtrc(A_el[8,10:20], A_el[9,10:20])

	B_11 = k_row[0:4]
	B_12 = k_row[4:8]
	B_13 = k_row[8:13]
	B_21 = l_row[0:4]
	B_22 = l_row[4:8]
	B_23 = l_row[8:13]
	B_31 = m_row[0:4]
	B_32 = m_row[4:8]
	B_33 = m_row[8:13]

	p_1 = pz4pz3(B_23, B_12) - pz4pz3(B_13, B_22)
	p_2 = pz4pz3(B_13, B_21) - pz4pz3(B_23, B_11)
	p_3 = pz3pz3(B_11, B_22) - pz3pz3(B_12, B_21)

	n_row = pz7pz3(p_1, B_31) + pz7pz3(p_2, B_32) + pz6pz4(p_3, B_33)

	n_row_scaled = n_row / n_row[0]

	e_val, _ = np.linalg.eig(np.vstack((-n_row_scaled[1:], np.hstack([np.eye(9), np.zeros((9,1))]))))

	# e_valで虚数を含む要素を排除
	e_val = e_val[np.isreal(e_val)].real
	m = e_val.shape[0]

	R_all = [[] for _ in range(m)]
	t_all = [[] for _ in range(m)]
	E_all = [[] for _ in range(m)]
	Eo_all = [[] for _ in range(m)]

	m = 0
	for z in e_val:
		p_z6 = np.array([z**6, z**5, z**4, z**3, z**2, z, 1], dtype=np.float64)
		p_z7 = np.insert(p_z6, 0, z**7)

		x = np.dot(p_1, p_z7) / np.dot(p_3, p_z6)
		y = np.dot(p_2, p_z7) / np.dot(p_3, p_z6)

		Eo = x * Xmat + y * Ymat + z * Zmat + Wmat
		Eo_all[m] = Eo
		U, _, Vt = np.linalg.svd(Eo)
		V = Vt.transpose()


		E = np.dot(np.dot(U, np.diag(np.array([1, 1, 0], dtype=np.float64))), Vt)
		E_all[m] = E

		#%stop here if nothing else is required to be computed
		#if nargout < 2
		#	m = m + 1;
		#	continue
		#end

		if np.linalg.det(U) < 0:
			U[:, 2] = -U[:, 2]

		if np.linalg.det(V) < 0:
			V[:, 2] = -V[:, 2]

		D = np.array([[ 0, 1, 0],
			          [-1, 0, 0],
			          [ 0, 0, 1]], dtype=np.float64)

		q_1 = q1[:, 0]
		q_2 = q2[:, 0]

		for n in range(4):
			if n == 0:
				t = U[:, 2]
				R = np.dot(np.dot(U, D), Vt)
			elif n == 1:
				t = -U[:, 2]
				R = np.dot(np.dot(U, D), Vt)
			elif n == 2:
				t = U[:, 2]
				R = np.dot(np.dot(U, D.transpose()), Vt)
			elif n == 3:
				t = -U[:, 2]
				R = np.dot(np.dot(U, D.transpose()), Vt)

			a = np.dot(E.transpose(), q_2)
			b = cross_vec3(q_1, np.append(a[0:2], 0))
			c = cross_vec3(q_2, np.dot(np.dot(np.diag(np.array([1, 1, 0], dtype=np.float64)), E), q_1))
			d = cross_vec3(a, b)

			P = np.hstack((R, t.reshape((3, 1))))
			C = np.dot(P.transpose(), c)
			Q = np.append(d * C[3], -np.dot(d[0:3], C[0:3]))

			if Q[2] * Q[3] < 0:
				continue

			c_2 = np.dot(P, Q)
			if (c_2[2] * Q[3] < 0):
				continue

			R_all[m] = R
			t_all[m] = t
			break

		m = m + 1

	return (E_all, R_all, t_all, Eo_all)

#function out = cross_vec3(u, v)
def cross_vec3(u, v):
	out = np.array([u[1]*v[2] - u[2]*v[1],
			        u[2]*v[0] - u[0]*v[2],
			        u[0]*v[1] - u[1]*v[0]], dtype=np.float64)
	return out

# function po = pz6pz4(p1, p2)
def pz6pz4(p1, p2):
	po = np.array([p1[0]*p2[0],
			       p1[1]*p2[0] + p1[0]*p2[1],
			       p1[2]*p2[0] + p1[1]*p2[1] + p1[0]*p2[2],
			       p1[3]*p2[0] + p1[2]*p2[1] + p1[1]*p2[2] + p1[0]*p2[3],
			       p1[4]*p2[0] + p1[3]*p2[1] + p1[2]*p2[2] + p1[1]*p2[3] + p1[0]*p2[4],
			       p1[5]*p2[0] + p1[4]*p2[1] + p1[3]*p2[2] + p1[2]*p2[3] + p1[1]*p2[4],
			       p1[6]*p2[0] + p1[5]*p2[1] + p1[4]*p2[2] + p1[3]*p2[3] + p1[2]*p2[4],
			       p1[6]*p2[1] + p1[5]*p2[2] + p1[4]*p2[3] + p1[3]*p2[4],
			       p1[6]*p2[2] + p1[5]*p2[3] + p1[4]*p2[4],
			       p1[6]*p2[3] + p1[5]*p2[4],
			       p1[6]*p2[4]], dtype=np.float64)
	return po

# function po = pz7pz3(p1, p2)
def pz7pz3(p1, p2):
	po = np.array([p1[0]*p2[0],
			       p1[1]*p2[0] + p1[0]*p2[1],
			       p1[2]*p2[0] + p1[1]*p2[1] + p1[0]*p2[2],
			       p1[3]*p2[0] + p1[2]*p2[1] + p1[1]*p2[2] + p1[0]*p2[3],
			       p1[4]*p2[0] + p1[3]*p2[1] + p1[2]*p2[2] + p1[1]*p2[3],
			       p1[5]*p2[0] + p1[4]*p2[1] + p1[3]*p2[2] + p1[2]*p2[3],
			       p1[6]*p2[0] + p1[5]*p2[1] + p1[4]*p2[2] + p1[3]*p2[3],
			       p1[7]*p2[0] + p1[6]*p2[1] + p1[5]*p2[2] + p1[4]*p2[3],
			       p1[7]*p2[1] + p1[6]*p2[2] + p1[5]*p2[3],
			       p1[7]*p2[2] + p1[6]*p2[3],
			       p1[7]*p2[3]], dtype=np.float64)
	return po

# function po = pz4pz3(p1, p2)
def pz4pz3(p1, p2):
	po = np.array([p1[0]*p2[0],
			       p1[1]*p2[0] + p1[0]*p2[1],
			       p1[2]*p2[0] + p1[1]*p2[1] + p1[0]*p2[2],
			       p1[3]*p2[0] + p1[2]*p2[1] + p1[1]*p2[2] + p1[0]*p2[3],
			       p1[4]*p2[0] + p1[3]*p2[1] + p1[2]*p2[2] + p1[1]*p2[3],
			       p1[4]*p2[1] + p1[3]*p2[2] + p1[2]*p2[3],
			       p1[4]*p2[2] + p1[3]*p2[3],
			       p1[4]*p2[3]], dtype=np.float64)
	return po

# function po = pz3pz3(p1, p2)
def pz3pz3(p1, p2):
	po = np.array([p1[0]*p2[0],
			       p1[0]*p2[1] + p1[1]*p2[0],
			       p1[0]*p2[2] + p1[1]*p2[1] + p1[2]*p2[0],
			       p1[0]*p2[3] + p1[1]*p2[2] + p1[2]*p2[1] + p1[3]*p2[0],
			       p1[1]*p2[3] + p1[2]*p2[2] + p1[3]*p2[1],
			       p1[2]*p2[3] + p1[3]*p2[2],
			       p1[3]*p2[3]], dtype=np.float64)
	return po

# function po = partial_subtrc(p1, p2)
def partial_subtrc(p1, p2):
	po = np.array([-p2[0], p1[0] - p2[1], p1[1] - p2[2], p1[2],
	               -p2[3], p1[3] - p2[4], p1[4] - p2[5], p1[5],
		           -p2[6], p1[6] - p2[7], p1[7] - p2[8], p1[8] - p2[9], p1[9]], dtype=np.float64)
	return po

# function B = gj_elim_pp(A)
def gj_elim_pp(A):
	_, _, U = scipy.linalg.lu(A)

	B = np.zeros((10,20))
	B[0:3,:] = U[0:3,:]

	B[9,:] = U[9,:] / U[9,9]
	B[8,:] = [U[8,:] - U[8,9]*B[9,:]] / U[8,8]
	B[7,:] = [U[7,:] - U[7,8]*B[8,:] - U[7,9]*B[9,:]] / U[7,7]
	B[6,:] = [U[6,:] - U[6,7]*B[7,:] - U[6,8]*B[8,:] - U[6,9]*B[9,:]] / U[6,6]
	B[5,:] = [U[5,:] - U[5,6]*B[6,:] - U[5,7]*B[7,:] - U[5,8]*B[8,:] - U[5,9]*B[9,:]] / U[5,5]
	B[4,:] = [U[4,:] - U[4,5]*B[5,:] - U[4,6]*B[6,:] - U[4,7]*B[7,:] - U[4,8]*B[8,:] - U[4,9]*B[9,:]] / U[4,4]

	return B

# function pout = p1p1(p1, p2)
def p1p1(p1, p2):
	return np.array([
		p1[0] * p2[0],
		p1[1] * p2[1],
		p1[2] * p2[2],
		p1[0] * p2[1] + p1[1] * p2[0],
		p1[0] * p2[2] + p1[2] * p2[0],
		p1[1] * p2[2] + p1[2] * p2[1],
		p1[0] * p2[3] + p1[3] * p2[0],
		p1[1] * p2[3] + p1[3] * p2[1],
		p1[2] * p2[3] + p1[3] * p2[2],
		p1[3] * p2[3]
	], dtype=np.float64)

# function pout = p2p1(p1,p2)
def p2p1(p1, p2):
	return np.array([
		p1[0] * p2[0],
        p1[1] * p2[1],
        p1[2] * p2[2],
        p1[0] * p2[1] + p1[3] * p2[0],
        p1[1] * p2[0] + p1[3] * p2[1],
        p1[0] * p2[2] + p1[4] * p2[0],
        p1[2] * p2[0] + p1[4] * p2[2],
        p1[1] * p2[2] + p1[5] * p2[1],
        p1[2] * p2[1] + p1[5] * p2[2],
        p1[3] * p2[2] + p1[4] * p2[1] + p1[5] * p2[0],
        p1[0] * p2[3] + p1[6] * p2[0],
        p1[1] * p2[3] + p1[7] * p2[1],
        p1[2] * p2[3] + p1[8] * p2[2],
        p1[3] * p2[3] + p1[6] * p2[1] + p1[7] * p2[0],
        p1[4] * p2[3] + p1[6] * p2[2] + p1[8] * p2[0],
        p1[5] * p2[3] + p1[7] * p2[2] + p1[8] * p2[1],
        p1[6] * p2[3] + p1[9] * p2[0],
        p1[7] * p2[3] + p1[9] * p2[1],
        p1[8] * p2[3] + p1[9] * p2[2],
        p1[9] * p2[3]
	], dtype=np.float64)

# function pts = triangulate(pts1, pts2, K1, K2, E, R, t)
def triangulate(pts1, pts2, K1, K2, E, R, t):
	if pts1.shape[0] != 2 or pts2.shape[0] != 2:
		print("warning: pts1 and pts2 must be of size 2xn")
		return None

	if pts1.shape != pts2.shape:
		print("warning: pts1 and pts2 must have the same number of points")
		return None

	if R.shape != (3, 3):
		print("warning: R must be of size 3x3")
		return None

	if t.shape != (3,):
		error("warning: t must be of size 3x1")
		return None

	n = pts1.shape[1]
	pts = np.zeros((4, n))
	K1_inv = np.array([[1.0/K1[0,0],0.0,-K1[0,2]/K1[0,0]],[0.0,1.0/K1[1,1],-K1[1,2]/K1[1,1]],[0,0,1]], dtype=np.float64)
	K2_inv = np.array([[1.0/K2[0,0],0.0,-K2[0,2]/K2[0,0]],[0.0,1.0/K2[1,1],-K2[1,2]/K2[1,1]],[0,0,1]], dtype=np.float64)
	q_1 = np.dot(K1_inv, np.vstack((pts1, np.ones((1, n), dtype=np.float64))))
	q_2 = np.dot(K2_inv, np.vstack((pts2, np.ones((1, n), dtype=np.float64))))

	for m in range(n):
		a = np.dot(E.transpose(), q_2[:, m])
		b = cross_vec3(q_1[:,m], np.append(a[0:2], 0))
		c = cross_vec3(q_2[:,m], np.dot(np.dot(np.diag(np.array([1, 1, 0], dtype=np.float64)), E), q_1[:,m]))
		d = cross_vec3(a, b)

		P = np.hstack((R, t.reshape((3, 1))))
		C = np.dot(P.transpose(), c)
		Q = np.append(d * C[3], -np.dot(d[0:3], C[0:3]))

		pts[:, m] = Q
	
	return pts
