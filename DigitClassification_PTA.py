# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 22:57:25 2018

@author: mayur
"""
import gzip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

IMAGE_SIZE = 28

def extract_data(filename, num_images):
    print('Extracting', filename)

    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data


def extract_labels(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels


# define step function
def step_function(num):
    for z in np.arange(len(num)):
        if num[z] >= 0:
            num[z] = 1
        else:
            num[z] = 0
    return num


# define  the  output  from  the  largest  value
def decide_out(array_v):
    max_index = 0
    temp = array_v[0]
    for z in np.arange(len(array_v)):
        if array_v[z] > temp:
            temp = array_v[z]
            max_index = z
    return max_index


# desired out as array
def get_desired_array(desired_number):
    dummy = np.zeros(10).reshape(10, 1)
    dummy[desired_number] = 1
    return dummy


# function to detect inequality in 2 arrays
def check_exact(a, b):
    err = 0
    for i in range(0, 10):
        for j in range(0, 784):
            if a[i][j] != b[i][j]:
                err = err + 1
    return err


train_data_filename = './train-images-idx3-ubyte.gz'
train_labels_filename = './train-labels-idx1-ubyte.gz'
test_data_filename = './t10k-images-idx3-ubyte.gz'
test_labels_filename = './t10k-labels-idx1-ubyte.gz'

# Extract  it  into  np  arrays.
train_data = extract_data(train_data_filename, 60000)
desired_output = extract_labels(train_labels_filename, 60000)
test_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)
# train_data[0][:, :, 0] ====> (28, 28)


# 3.1
w = np.random.uniform(-1, 1, [10, 784])
elements = 60000  # elements  <=  600000
epsilon = 0.130
eta = 0.01
epoch = 0
errors = []
actual_out = []
not_good_ratio = True


# 3.2
while not_good_ratio:
    # 3.1.1
    errors.append(0)
    for i in range(0, elements):
        # 3.1.1.1
        x = np.reshape(train_data[i][:, :, 0], (784, 1))  # (784,  1)
        # Single Sample
        v = np.dot(w, x)  # v  =  [v0  v1  v2  ....  v9] 107
        #  3.1.1.2
        actual_out.append(decide_out(v))  # actual_out[0] ===> X0 109	# 3.1.1.3
        if actual_out[i] != desired_output[i]:
            errors[epoch] += 1
    actual_out = []
    # ----------------------------------------------------------------------
    w1 = w
    # 3.1.2
    epoch = epoch + 1
    # 3.1.3
    for i in range(0, elements):  # update  weights
        x = np.reshape(train_data[i][:, :, 0], (784, 1))
        v = np.dot(w, x)
        a = get_desired_array(desired_output[i])  # (10, 1)
        b = step_function(v)  # (10, 1)
        o = (a - b)
        c = eta * o
        k = np.add(w, np.dot(c, x.T))  # (10,1)*(1,  784)=(10,  784)
        w = k
    # ------------------------------------------------------------
    w2 = w
    # print("check  exact:  {}".format(check_exact(w1,  w2)))
    ratio = errors[epoch - 1] / elements
    print("epoch# ", epoch)
    print("ratio:  {}".format(ratio))
    print(errors)
    if ratio <= epsilon:
        not_good_ratio = False

plt.figure(1)
sns.set()
y = errors
N = len(y)
x = range(N)
plt.plot(x, y, label='errors  vs  epoch')
plt.ylabel('Number  of  errors')
plt.xlabel('Epoch  Number')
plt.title('Eta  =  0.01  ,  n  =  60000,  epsilon  =  0.130')
plt.legend()
plt.show()

# (e)
err = 0
actual_out = []
for i in range(0, 10000):
    # ---------------------------------------------------
    x = np.reshape(test_data[i][:, :, 0],(784, 1))  # Single  Sample
    v = np.dot(w,  x)  # v  =  [v0  v1  v2  ....  v9]
    # ---------------------------------------------------------
    actual_out.append(decide_out(v))
    # --------------------------------------------------------------
    if actual_out[i] != test_labels[i]:
        err = err + 1
print("miss  classified  samples  ={}".format(err))

# Function  for  plotting  -----------------------------------------------
# plt.imshow(train_data[0][:,  :,  0],  cmap='Greys',  interpolation='nearest')
# plt.show()


