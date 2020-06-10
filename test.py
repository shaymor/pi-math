from matplotlib import pyplot as plt
import imageio


im = imageio.imread("process/image.png")
plt.imshow(im)
plt.show()