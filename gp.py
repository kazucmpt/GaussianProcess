import numpy as np 
import matplotlib.pyplot as plt
import math
import dataset

xmin = -7
xmax = 5
noise_level = 0.1

class GaussianProcess:

	def __init__(self, theta1, theta2, theta3):
		# thetax are used in rbf kernel elements
		self.theta1 = theta1
		self.theta2 = theta2
		self.theta3 = theta3
		self.xtrain = None
		self.ytrain = None
		self.xtest = None
		self.ytest = None
		self.kernel_inv = None # matrix
		self.mean_arr = None
		self.var_arr = None

	def rbf(self, p, q):
		return self.theta1 * math.exp(- (p-q)**2 / self.theta2)

	def train(self, data):
		self.xtrain = data.x
		self.ytrain = data.y
		N = len(self.ytrain)

		kernel = np.empty((N, N))
		for n1 in range(N):
			for n2 in range(N):
				kernel[n1, n2] = self.rbf(self.xtrain[n1], self.xtrain[n2]) + chr(n1, n2)*self.theta3

		self.kernel_inv = np.linalg.inv(kernel)

	def test(self, test_data):
		self.xtest = test_data.x

		N = len(self.xtrain)
		M = len(self.xtest)	
		
		partial_kernel_train_test = np.empty((N, M))
		for m in range(M):
			for n in range(N):
				partial_kernel_train_test[n, m] = self.rbf(self.xtrain[n], self.xtest[m])
		
		partial_kernel_test_test = np.empty((M, M))
		for m1 in range(M):
			for m2 in range(M):
				partial_kernel_test_test[m1, m2] = self.rbf(self.xtest[m1], self.xtest[m2]) + chr(m1, m2)*self.theta3
		
		self.mean_arr = partial_kernel_train_test.T @ self.kernel_inv @ self.ytrain
		self.var_arr  = partial_kernel_test_test - partial_kernel_train_test.T @ self.kernel_inv @ partial_kernel_train_test

def draw(train_data, test_data, no_noise_data, mean_arr, var_arr):	
	xtest = test_data.x
	xtrain = train_data.x
	ytrain = train_data.y

	var_arr = np.diag(var_arr)

	boundary_upper = np.empty(len(xtest))
	boundary_lower = np.empty(len(xtest))
	for i in range(len(xtest)):
		boundary_upper[i] = mean_arr[i] + var_arr[i]
		boundary_lower[i] = mean_arr[i] - var_arr[i]

	plt.figure(figsize=(15, 10))
	plt.scatter(xtrain, ytrain, label="Train Data", color="red")
	plt.plot(xtest, mean_arr, label="Predicted Line by limted test data")
	plt.plot(no_noise_data.x, no_noise_data.y, label="GT without noise", color="black", linestyle='dashed')
	plt.xlabel("x", fontsize=16)
	plt.ylabel("y", fontsize=16)
	plt.xlim(xmin, xmax)
	plt.ylim(-0.5,3.5)
	plt.tick_params(labelsize=16)
	plt.title("Predicted line by Gaussian Process", fontsize="16")
	plt.fill_between(xtest, boundary_upper, boundary_lower, facecolor='y',alpha=0.3)
	plt.legend(["Train Data", "Predicted Line by limted test data", "GT without noise", "confidence interval $\pm\sigma$"],fontsize=16)
	plt.savefig("gp.png")
	plt.show()

def chr(a, b):
	if a == b:
		return 1
	else:
		return 0

def main():
	train_data = dataset.DataSet(xmin, xmax, num_data=50, noise_level=noise_level)
	test_data = dataset.DataSet(xmin, xmax, num_data=1000, noise_level=noise_level)
	no_noise_data = dataset.DataSet(xmin, xmax, num_data=1000, noise_level=0.0)

	model = GaussianProcess(theta1=1, theta2=0.4, theta3=0.1)
	model.train(train_data)
	model.test(test_data)

	draw(train_data, test_data, no_noise_data, model.mean_arr, model.var_arr)

if __name__ == '__main__':
	main()
