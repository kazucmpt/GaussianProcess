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

	def train(self, train_data):
		x = train_data.x
		y = train_data.y
		N = len(x)
		self.design_matrix = np.empty((N, self.num_basis))
		for i in range(N):
			for j in range(self.num_basis):
				self.design_matrix[i,j] = self.basis(x[i], self.mus[j])

		self.weights = np.linalg.inv(self.design_matrix.transpose() @ self.design_matrix) @ self.design_matrix.transpose() @ y

	def predict(self):
		x = np.linspace(-7, 5, 100)
		self.pre_y = np.zeros(len(x))
		for i in range(len(x)):
			for j in range(self.num_basis):
				self.pre_y[i] += self.weights[j] * self.basis(x[i], self.mus[j])

def draw(pre_y, no_noise_data=None, train_data=None):
	x = np.linspace(-7, 5, 100)

	plt.figure(figsize=(15, 10))	
	if train_data != None:
		plt.scatter(train_data.x, train_data.y, label="input data with noise")
	if no_noise_data != None:
		plt.plot(no_noise_data.x, no_noise_data.y, label="GT without noise", color="black", linestyle='dashed')

	plt.xlabel("x", fontsize=16)
	plt.ylabel("y", fontsize=16)
	plt.xlim(-7,5)
	plt.ylim(-0.5,3.5)
	plt.tick_params(labelsize=16)
	plt.title("Predicted line by RBFR", fontsize="16")
	plt.plot(x, pre_y, label="predicted line", color="tomato")
	plt.legend(fontsize=16)
	plt.savefig("rbfr.png")
	plt.show()

def main():
	xmin = -7
	xmax = +5
	noise_level = 0.1

	train_data = dataset.DataSet(xmin, xmax, num_data=20, noise_level=noise_level)
	no_noise_data = dataset.DataSet(xmin, xmax, num_data=1000, noise_level=0.0)
	#train_data.plot()

	model = RadialBasisFunctionRegression(num_basis=10)
	model.train(train_data)
	model.predict()
	draw(model.pre_y, no_noise_data, train_data)

if __name__ == '__main__':
	main()
