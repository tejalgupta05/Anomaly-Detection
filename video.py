from gaussian import estimateGaussian, multivariateGaussian
import numpy as np
import cv2

"""
stream = cv2.VideoCapture(0)

while True:
    success, frame = stream.read()

    img = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_LINEAR)
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    mean, variance = estimateGaussian(img)
    p = multivariateGaussian(img, mean, variance)

    probability = np.mean(p)
    print(probability < epsilon)
    
    #cv2.imwrite("img.jpeg", frame)
    cv2.imshow("Image", frame)
    cv2.waitKey(1)
"""

epsilon = 1.0239287920205515e-07
anomaly = 0

for i in range(1, 130):
    img = cv2.imread(f"D:\\ML\\Mini Project\\images\\1\\Frame {i}.jpg")
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_LINEAR)
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mean, variance = estimateGaussian(img)
    p = multivariateGaussian(img, mean, variance)

    probability = np.mean(p)

    if probability < epsilon:
        anomaly += 1

print(anomaly/129)
