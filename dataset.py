import numpy as np 
import matplotlib.pyplot as plt
import math

class DataSet:

	def __init__(self, xmin, xmax, num_data, noise_level):
		self.x = (xmax - xmin) * np.random.rand(num_data) + xmin
		self.y = np.empty(num_data)

		for i in range(num_data):
			self.y[i] = self.make_y(self.x[i], noise_level)

		self.x, self.y = self.dual_sort(self.x, self.y)

	@staticmethod
	def make_y(x, noise_level):
			if x > 0:
				return 1.0/2.0*x**(1.2) + math.cos(x) + np.random.normal(0, noise_level)
			else:
				return math.sin(x) + 1.1**x + np.random.normal(0, noise_level)

	@staticmethod
	def dual_sort(p, q):
		tmp = np.argsort(p)
		q = q[tmp]
		p = np.sort(p)

		return p, q

	def plot(self):
		plt.scatter(self.x, self.y)
		plt.show()
