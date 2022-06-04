from gaussian import estimateGaussian, multivariateGaussian, selectThreshold
from matplotlib import pyplot as plt
from matplotlib.image import imsave
from scipy import stats
import numpy as np
import cv2


def run():
    probablities = []
    y = []

    for i in range(1, 939):
        img = cv2.imread(f"D:\\ML\\Mini Project\\images\\0\\Frame {i}.jpg")

        ratio = 50 / img.shape[0]
        dim = (int(img.shape[1]*ratio), 50)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        #print(img.shape)
        X = img.reshape((img.shape[0] * img.shape[1], 3))

        mean, variance = estimateGaussian(X)
        p = multivariateGaussian(X, mean, variance)
        
        frequent = stats.mode(p)
        #print(frequent[0], frequent)
        #mode = stats.mode()
        #print(p.shape)

        probablities.append(np.mean(p))
        y.append(0)
    
    for i in range(1, 130):
        im1 = cv2.imread(f"D:\\ML\\Mini Project\\images\\1\\Frame {i}.jpg").resize((150, 150))
        pixel_mat = np.asfarray(im1)
        X = pixel_mat.reshape((pixel_mat.shape[0] * pixel_mat.shape[1], 3))

        mean, variance = estimateGaussian(X)
        p = multivariateGaussian(X, mean, variance)

        probablities.append(np.mean(p))
        y.append(1)


    probablities = np.array(probablities).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    epsilon, F1 = selectThreshold(probablities, y)

    print(probablities < epsilon)

    return epsilon, F1
    """
    plt.plot(probablities)
    plt.scatter(200, epsilon)
    plt.show()


    plt.scatter(pixel_mat[:, 0], pixel_mat[:, 1])
    plt.show()
    ax = plt.axes(projection='3d')
    ax.scatter3D(pixel_mat[:2000, 0], pixel_mat[:2000, 1], pixel_mat[:2000, 2], 'gray')

    plt.show()
    """

""" 
im1 = cv2.imread(f"D:\\ML\\Mini Project\\images\\1\\Frame 1.jpg")
im1 = cv2.resize(im1, (150, 150))
cv2.imshow("img", im1)
cv2.waitKey(0)
exit()
"""

epsilon, F1 = run()
print(epsilon, F1)
