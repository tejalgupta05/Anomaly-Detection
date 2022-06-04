import matplotlib.pyplot as plt
import cv2


source = r"D:\ML\Mini Project\Test3\1\2.jpg"

image = cv2.imread(source)
image = cv2.resize(image, (30, 30), interpolation=cv2.INTER_LINEAR)
#cv2.imshow("image", image)
#cv2.waitKey(0)


ax = plt.axes(projection='3d')
ax.scatter3D(image[:, 0], image[:, 1], image[:, 2], 'gray')
plt.show()

"""
resize to 10x10 rgb
find the images with anomaly from data by looking at the indices of anomalous pixel examples in the map
"""
