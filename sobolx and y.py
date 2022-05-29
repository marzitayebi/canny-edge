import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/home/marzi/Documents/python/images/boz.jpg", cv2.IMREAD_GRAYSCALE)
lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3) #cv2.cv_64f is data type
lap = np.uint8(np.absolute(lap))
sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
edges = cv2.Canny(img, 100, 200) 

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX, sobelY)

titles = ["image", "Laplacian", "sobelX", "sobelY",  "sobelCombined", ]
images = [img, lap, sobelX, sobelY, sobelCombined]



for i in range(5):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], "gray")
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

cv2.imshow ("image", img)
cv2.imshow ("sobelCombined", sobelCombined)
cv2.imshow ("sobelY", sobelY)
cv2.imshow ("Canny", edges)
cv2.imshow ("sobelX", sobelX)
cv2.imshow ("sobelX", sobelX)

plt.imshow(img)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
