import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

pic = Image.open('cover.jpg')
pic_arr = np.asanyarray(pic)

plt.figure(1)
plt.imshow(pic_arr)
plt.show()

plt.show(pic[:,:, 0], cmap='gray')
