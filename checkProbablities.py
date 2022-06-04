from audioop import mul
from gaussian import estimateGaussian, multivariateGaussian, selectThreshold
from matplotlib import pyplot as plt
import numpy as np
import cv2


probablities = []
y = []

for i in range(1, 939):
    image = cv2.imread(f"D:\\ML\\Mini Project\\images\\0\\Frame {i}.jpg")
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    X = image.reshape((image.shape[0] * image.shape[1], 3))

    mean, variance = estimateGaussian(X)
    p = multivariateGaussian(X, mean, variance)
    
    probablities.append(p)
    y.append(0)


for i in range(1, 130):
    image = cv2.imread(f"D:\\ML\\Mini Project\\images\\1\\Frame {i}.jpg")
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    X = image.reshape((image.shape[0] * image.shape[1], 3))

    mean, variance = estimateGaussian(X)
    p = multivariateGaussian(X, mean, variance)

    probablities.append(p)
    y.append(1)


probablities = np.array(probablities)
y = np.array(y).reshape(-1, 1)

mean_prob, var_prob = estimateGaussian(probablities)
prob = multivariateGaussian(probablities, mean_prob, var_prob)

print(prob)

print("selecting")
epsilon, F1 = selectThreshold(prob, y)

print(F1, epsilon)



"""
plt.subplot(2, 2, 1)
plt.plot(probablities[:, 0])

plt.subplot(2, 2, 2)
plt.plot(probablities[:, 1])

plt.subplot(2, 2, 3)
plt.plot(probablities[:, 2])

plt.subplot(2, 2, 4)
plt.plot(probablities[:, 3])

plt.show()
"""