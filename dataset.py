import numpy as np 
import matplotlib.pyplot as plt
import math

class DataSet:

	def __init__(self, xmin, xmax, num_data, noise_level):
		self.xmin = xmin
		self.xmax = xmax
		self.x = (xmax - xmin) * np.random.rand(num_data) + xmin
		self.x = np.sort(self.x)
		
		self.y = np.empty(num_data)
		for i in range(num_data):
			self.y[i] = self.make_y(self.x[i], noise_level)

	@staticmethod
	def make_y(x, noise_level):
			if x > 0:
				return 1.0/2.0*x**(1.2) + math.cos(x) + np.random.normal(0, noise_level)
			else:
				return math.sin(x) + 1.1**x + np.random.normal(0, noise_level)

	def plot(self):	
		plt.figure(figsize=(15, 10))
		plt.xlabel("x", fontsize=16)
		plt.ylabel("y", fontsize=16)
		plt.xlim(self.xmin, self.xmax)
		plt.tick_params(labelsize=16)
		plt.scatter(self.x, self.y)
		plt.show()
