from operator import mul
from numpy.core.fromnumeric import shape
from gaussian import estimateGaussian, multivariateGaussian, selectThreshold
from matplotlib import pyplot as plt
import numpy as np
import cv2


probablities = []
y = []

pixel_arr = []

for i in range(1, 10):
    img = cv2.imread(f"D:\\ML\\Mini Project\\images\\0\\Frame {i}.jpg")
    img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_LINEAR)

    X = np.reshape(img, (900, 3))
    pixel_arr.append(X)

    mean, variance = estimateGaussian(X)
    p = multivariateGaussian(X, mean, variance)

    probablities.append(p)
    y.append(0)


for i in range(1, 3):
    img = cv2.imread(f"D:\\ML\\Mini Project\\images\\1\\Frame {i}.jpg")
    img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_LINEAR)
    
    X = np.reshape(img, (900, 3))
    pixel_arr.append(X)

    mean, variance = estimateGaussian(X)
    p = multivariateGaussian(X, mean, variance)

    probablities.append(p)
    y.append(1)


pixel_arr = np.array(pixel_arr)
pixel_arr = pixel_arr.reshape((11*900, 3))

probablities = np.array(probablities)

for row in probablities:
    print(row)

exit()

plt.plot(probablities)
plt.show()

probablities = np.array(probablities).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

epsilon, F1 = selectThreshold(probablities, y)

print(probablities < epsilon)
print(probablities)
print(epsilon, F1)



plt.scatter(pixel_mat[:, 0], pixel_mat[:, 1])
plt.show()
ax = plt.axes(projection='3d')
ax.scatter3D(pixel_mat[:2000, 0], pixel_mat[:2000, 1], pixel_mat[:2000, 2], 'gray')
plt.show()
