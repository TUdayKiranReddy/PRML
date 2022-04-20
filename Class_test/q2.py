import numpy as np

A = np.array([[0.1, 0.3, 0.3, 0.3], [0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25], [0.1, 0.4, 0.4, 0.1]])
B = np.array([[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.6, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2], [0, 0, 0, 0.5, 0.5]])
pi= np.array([0.25, 0.25, 0.25, 0.25]).T

O = [4, 2, 1, 0]

# def alpha(t, i):
# 	if t == 1:
# 		return pi[i, 0]*B[i, O[t-1]]

# 	s = 0
# 	for j in range(pi.shape[0]):
# 		s += alpha(t-1, j)*A[j, i]*B[i, O[t-1]]
# 	return s

# print(np.sum([alpha(4, i) for i in range(4)]))

# def delta(t, i):
# 	if t == 1:
# 		return pi[i, 0]*B[i, O[t-1]]
# 	arr = [delta(t-1, j)*A[j, i] for j in range(A.shape[0])]
# 	return np.max(arr)*B[i, O[t-1]]

# def psi(t, i):
# 	if t == 1:
# 		return 0
# 	return np.argmax([delta(t-1, j)*A[j, i] for j in range(A.shape[0])])

# ds = [delta(4, k) for k in range(4)]
# print(ds)

# qT = np.argmax(ds)
# for i in reversed(range(4)):
# 	print(i+1, qT)
# 	qT = psi(i+1, qT)
N = A.shape[0]
M = B.shape[1]
T = 4

alpha = np.zeros((N, T))

alpha[:, 0] = pi * B[:, O[0]]
for i in range(1, T):
	alpha[:, i] = A.T@alpha[:, i-1] * B[:, O[i]]

print("Alpha = \n", alpha)

print("P(O/l) = \n", np.sum(alpha[:, -1]))

beta = np.zeros((N, T))

beta[:, -1] = 1

for i in reversed(range(T-1)):
	beta[:, i] = A@(beta[:, i+1] * B[:, O[i]])

print("Beta = \n", beta)

p = np.sum(pi * beta[:, 0] * B[:, O[0]])
print(p)