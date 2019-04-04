import numpy as np 
import matplotlib.pyplot as plt
import math
import dataset

class GaussianProcess:

	def __init__(self, theta1, theta2, theta3):
		self.theta1 = theta1
		self.theta2 = theta2
		self.theta3 = theta3

	def rbf(self, p, q):
		if p == q :
			return self.theta1 * math.exp(- (p-q)**2 / self.theta2) + self.theta3
		else:
			return self.theta1 * math.exp(- (p-q)**2 / self.theta2)

	def train(self, train_data):
		self.xtrain = train_data.x
		self.ytrain = train_data.y
		N = len(self.ytrain)

		self.kernel = np.empty((N, N))
		for n1 in range(N):
			for n2 in range(N):
				self.kernel[n1, n2] = self.rbf(self.xtrain[n1], self.xtrain[n2])

	def test(self, test_data):
		self.xtest  = test_data.x

		N = len(self.xtrain)
		M = len(self.xtest)	
		
		partial_kernel_train_test = np.empty((N, M))
		for m in range(M):
			for n in range(N):
				partial_kernel_train_test[n, m] = self.rbf(self.xtrain[n], self.xtest[m])
		
		partial_kernel_test_test = np.empty((M, M))
		for m1 in range(M):
			for m2 in range(M):
				partial_kernel_test_test[m1, m2] = self.rbf(self.xtest[m1], self.xtest[m2])
		
		
		self.mean = partial_kernel_train_test.T @ np.linalg.inv(self.kernel) @ self.ytrain
		self.var  = partial_kernel_test_test - partial_kernel_train_test.T @ np.linalg.inv(self.kernel) @ partial_kernel_train_test

	def plot_predict(self):	
		var = np.diag(self.var)
		self.xtest, mean, var = tiplet_shuffle(self.xtest, self.mean, var)

		boundary_upper = np.empty(len(self.xtest))
		boundary_lower = np.empty(len(self.xtest))
		for i in range(len(self.xtest)):
			boundary_upper[i] = mean[i] + var[i]
			boundary_lower[i] = mean[i] - var[i]

		plt.scatter(self.xtrain, self.ytrain, label="Train Data", color="red")
		plt.plot(self.xtest, mean, label="Test Data")
		plt.legend(fontsize=16)
		plt.xlabel("x", fontsize=16)
		plt.ylabel("y", fontsize=16)
		plt.xlim(-7,5)
		plt.ylim(-0.5,3.5)
		plt.title("Predicted line by Gaussian Process", fontsize="16")
		plt.fill_between(self.xtest, boundary_upper, boundary_lower, facecolor='y',alpha=0.3)
		plt.show()

def tiplet_shuffle(p, q, r):
	tmp = np.argsort(p)
	q = q[tmp]
	r = r[tmp]
	p = np.sort(p)

	return p, q, r

def main():
	train_data = dataset.DataSet(-7, 5, num_data=50, noise_level=0.1)
	test_data = dataset.DataSet(-7, 5, num_data=150, noise_level=0.1)

	model = GaussianProcess(theta1=1, theta2=0.4, theta3=0.1)
	model.train(train_data)
	model.test(test_data)
	model.plot_predict()

if __name__ == '__main__':
	main()
