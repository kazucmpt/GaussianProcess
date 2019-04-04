import numpy as np 
import matplotlib.pyplot as plt
import math
import dataset
#Radial basis function regression

class RadialBasisFunctionRegression:

	def __init__(self, num_basis):
		self.num_basis = num_basis
		self.weights = np.empty(self.num_basis)
		self.mus = np.linspace(-7, 5, self.num_basis)

	@staticmethod
	def basis(x, mu, sigma=1):
		return math.exp(-(x-mu)**2/sigma**2)

	def train(self, x, y):
		N = len(x)
		self.design_matrix = np.empty((N, self.num_basis))
		for i in range(N):
			for j in range(self.num_basis):
				self.design_matrix[i,j] = self.basis(x[i], self.mus[j])

		self.weights = np.linalg.inv(self.design_matrix.transpose() @ self.design_matrix) @ self.design_matrix.transpose() @ y

	def plot_predict_curve(self, train_data=None):
		x = np.linspace(-7, 5, 100)
		pre_y = np.zeros(len(x))
		for i in range(len(x)):
			for j in range(self.num_basis):
				pre_y[i] += self.weights[j] * self.basis(x[i], self.mus[j])
	
		if train_data != None:
			plt.scatter(train_data.x, train_data.y, label="input data")
			plt.legend()

		plt.xlabel("x", fontsize=16)
		plt.ylabel("y", fontsize=16)
		plt.xlim(-7,5)
		plt.ylim(-0.5,3.5)
		plt.title("Predicted line by RBFR", fontsize="16")
		plt.plot(x, pre_y, color="tomato")
		plt.show()

def main():
	train_data = dataset.DataSet(-7, 5, num_data=20, noise_level=0.2)
	#train_data.plot()

	model = RadialBasisFunctionRegression(num_basis=10)
	model.train(train_data.x, train_data.y)
	model.plot_predict_curve(train_data)

if __name__ == '__main__':
	main()
