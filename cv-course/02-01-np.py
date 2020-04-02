import numpy as np

mylist = [1,2,3]
print(type(mylist))

list
print(np.array(mylist))
np.ndarray
print(np.arange(0,10,2))
print(np.zeros(shape=(5,5)))
np.random.seed(101)
arr = np.random.randint(0,100,10)
print (arr)
print(arr.max())
print(arr.argmax())
print(arr.reshape(2,5))
# arr = np.arange(0,100).reshape(5.20)
print(np.arange(0,100,).reshape(10,10))
arr = np.arange(0,100,).reshape(10,10)
print(arr[:,1].reshape(10,1))
print(arr[0:3, 0:4])