from gaussian import estimateGaussian, multivariateGaussian, selectThreshold
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


im1 = Image.open(f"D:\\ML\\Mini Project\\0\\1.jpg").resize((300, 300))
pixel_mat = np.asfarray(im1)
X = pixel_mat.reshape((pixel_mat.shape[0] * pixel_mat.shape[1], 3))

mean, variance = estimateGaussian(X)
p = multivariateGaussian(X, mean, variance)

ax = plt.axes(projection='3d')
ax.scatter3D(pixel_mat[:2000, 0], pixel_mat[:2000, 1], pixel_mat[:2000, 2], 'gray')
plt.show()
exit()
plt.plot(p)
plt.show()