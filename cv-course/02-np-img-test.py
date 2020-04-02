import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


print(np.ones(shape=(5,5))*10)

np.random.seed(101)
print(np.random.randint(0,100,(5,20)))
print(np.max(np.random.randint(0,100,(5,20))))
print(np.min(np.random.randint(0,100,(5,20))))

plt.figure(1)
plt.imshow(Image.open('cover.jpg'))
plt.show()

pic_arr = np.asarray(Image.open('cover.jpg'))

copy = pic_arr.copy()
copy[:,:,0]=0
plt.imshow(copy)
plt.show()
copy[:,:,1]=0
plt.imshow(copy)
plt.show()