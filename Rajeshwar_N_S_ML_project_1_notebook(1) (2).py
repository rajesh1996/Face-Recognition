
# coding: utf-8

# # Convert to Images, Train-Test split, Saving Images to Folders

# In[3]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import scipy
import scipy.io
import pathlib
from IPython.display import display
import imageio
from PIL import Image
from collections import defaultdict
import math


# In[4]:


#set working directory. insert your directory here
wd = '/home/rajesh/Documents/Rajeshwar_N_S_ML_Proj_1'
os.chdir(wd)


# In[7]:


#Load Dataset (mat file) using scipy function
data = scipy.io.loadmat(wd + '/Dataset/illumination.mat')


# In[8]:


#Convert into Images 40x48 and rotate it to display 
def convert_to_img(vector_img):
    img = vector_img.reshape(40,48)
    return Image.fromarray(img).rotate(270,expand=True) #preserve the entire image while rotating


# In[9]:


#saving images for each person (0-67) and their 21 illuminations (0-21 for each person)

""" The file 'illumination.mat' has a variable "illum" of size 1920x21x68.
reshape(illum(:, i,j), 48, 40) gives i^th image of j^th subject.
"""
        #create temp directories and put images in them
image_mat = data['illum']
for i in range(image_mat.shape[1]):
    for j in range(image_mat.shape[2]):

        face, illumn = '{0:0=2d}'.format(j), '{0:0=2d}'.format(i)
        pathlib.Path(wd+'/Dataset/illumination_tmp/face{0}'.format(face)).mkdir(parents=True, exist_ok=True)
        img = convert_to_img(image_mat[:,i,j])
        illum_name = 'face{0}_illumn{1}.tif'.format(face,illumn)
        img.save(wd+'/Dataset/illumination_tmp/face{0}/{1}'.format(face,illum_name))


# In[13]:


#Train-test split 70 -training, 30- test
random.seed(100)
train_test_split = 0.7 
train = []
test = []
subdir = wd+'/Dataset/illumination_tmp/'
faces = next(os.walk(subdir))[1]  # To loop through all the person folders

for illums in faces:
    filenames = os.listdir(subdir + illums)
    image_names = [filename for filename in filenames]
    image_paths = [subdir + illums + os.sep + image_name for image_name in image_names]
    random.shuffle(image_paths) # shuffe data
    train = train + image_paths[:int(train_test_split*len(image_names))]
    test = test + image_paths[int(train_test_split*len(image_names)):]


# In[14]:


print("train=",len(train))
print("test=",len(test))


# In[15]:


#Display an Image
def load_image(img):
    return imageio.imread(img).flatten(order='F')  # column-major flatten

img = load_image(test[10])
img = convert_to_img(img)
plt.imshow(img)
plt.show()


# In[16]:


#Pre-processing- Averaging. -Add up all the pixel values divided by the total length
def avg_image(train_data):
    average_image = [0]
    for filename in train_data:
        average_image += load_image(filename)
    average_image /= len(train_data)
    return average_image


# In[17]:



average_train = avg_image(train)
def mean_image(img):
    img = load_image(img)
    return img-average_train


# In[18]:


#create train and test directory seperatly for convinience
pathlib.Path(wd+'/final_data/train').mkdir(parents=True, exist_ok=True) 
pathlib.Path(wd+'/final_data/test').mkdir(parents=True, exist_ok=True)


# In[19]:


train[1]
chunk = train[1].split('/')
print(chunk)


# In[20]:


#Saving training images
for face in train:
    face_no = face.split('/') #extracting face number (saving in similar format as illumination)
    pathlib.Path(wd+'/final_data/train/' + face_no[-2]).mkdir(parents=True, exist_ok=True)
    filepath = '/'.join([wd+"/final_data/train"] + face_no[-2:])
    img = convert_to_img(mean_image(face))
    img.save(filepath)


# In[21]:


#Saving test Images
for face in test:
    face_no = face.split('/') #extracting face number (saving in similar format as illumination)
    pathlib.Path(wd+'/final_data/test/' + face_no[-2]).mkdir(parents=True, exist_ok=True)
    filepath = '/'.join([wd+"/final_data/test"] + face_no[-2:])
    img = convert_to_img(mean_image(face))
    img.save(filepath)


# In[22]:


Train_data = wd+'/final_data/train/'
Test_data = wd+'/final_data/test/'


# In[24]:


#code snippet to access the names of face folders and illuminations of trainset
illum_dir_train = []
face_dir_train = []
import os
for root, dirs, files in os.walk(Train_data):
   for name in files:
      illum_dir_train.append(name)
   for name in dirs:
      face_dir_train.append(os.path.join(root, name))
#illum_dir_train


# In[25]:


#face and illuminations of test set
illum_dir_test = []
face_dir_test = []
import os
for root, dirs, files in os.walk(Test_data):
   for name in files:
      illum_dir_test.append(name)
   for name in dirs:
     face_dir_test.append(os.path.join(root, name))


# In[26]:


#face_dir_test


# In[27]:


#Extracting the name of face from the filepath for labeling
chunk = face_dir_train[0].split('train/')
print("Example \n",chunk[-1])

chunk_2 = illum_dir_train[1].split('_')
print(chunk_2[-2])

#chunk[-1]==chunk_2[-2]


# In[23]:


"""for name in parent:
    chunk = name.split('train/')
    for name_1 in links:
        chunk_2 = name_1.split('_')
        if(chunk[-1]==chunk_2[-2]):
            img = imageio.imread(name+ '/' +name_1).flatten(order='F')
        else:
            continue"""
        


# In[28]:


#storing images in a list in order 
img = []
for name in face_dir_train:
    chunk = name.split('train/')
    for name_1 in illum_dir_train:
        chunk_2 = name_1.split('_')
        if(chunk[-1]==chunk_2[-2]):
            img.append(imageio.imread(name+ '/' +name_1).flatten(order='F'))
        else:
            continue
        


# In[29]:



for i in img:
    img_1 = convert_to_img(i)
    plt.imshow(img_1)
    plt.show()


# In[30]:


avg = [0]
for i in img:
    avg += i
img_1 = convert_to_img(avg)
    
plt.imshow(img_1)
plt.show()
print(img[0].shape)


# In[31]:


#Create a list of train x and train y. Slice the filepath using split function to get the training label as shown in an example before.
train_X = []
train_Y = []
for name in face_dir_train:
    chunk = name.split('train/')
    for name_1 in illum_dir_train:
        chunk_2 = name_1.split('_')
        if(chunk[-1]==chunk_2[-2]):
            train_X.append(imageio.imread(name+ '/' +name_1).flatten(order='F'))
            train_Y.append(chunk_2[-2])


# In[32]:


#similarly do it for test
test_X = []
test_Y = []
for name in face_dir_test:
    chunk = name.split('test/')
    for name_1 in illum_dir_test:
        chunk_2 = name_1.split('_')
        if(chunk[-1]==chunk_2[-2]):
            test_X.append(imageio.imread(name+ '/' +name_1).flatten(order='F'))
            test_Y.append(chunk_2[-2])


# In[33]:


#list of images
#train_X


# In[34]:


#test_X


# # KNN

# In[35]:


#Define func for euclidian
def euclideanDistance(vec1, vec2):
    distance = 0
    for x in range(len(vec1)):
        distance = distance + math.pow((vec1[x] - vec2[x]), 2)
    return math.sqrt(distance)


# In[36]:


# calculate the eulcidian distance between each record in the dataset to the new piece of data

#Then,sort all of the records in the training dataset by their distance to the new data. select the top k to return as the most similar neighbors.


from queue import PriorityQueue

def get_Neighbors(train_X, train_Y, test_sample, k_neighbours):
    distances = PriorityQueue()
    for i in range(len(train_X)):
       distances.put((euclideanDistance(test_sample, train_X[i]), train_Y[i]))
    neighbors = []
    for _ in range(k_neighbours):
        neighbors.append(distances.get())
    return neighbors

def maximum_votes(neighbors):
    votes = defaultdict(int)
    for i in range(len(neighbors)):
        label = neighbors[i][-1]
        votes[label] += 1
    return sorted(votes.items(), key=lambda k_v: k_v[1])[-1][0]


# In[40]:


get_ipython().run_cell_magic('time', '', '#Fit the data on KNN classifer defined above\n\npredicted = []\nfor i in range(len(test_X)):\n    neighbours = get_Neighbors(train_X,train_Y,test_X[i],1)\n    predicted.append(maximum_votes(neighbours))')


# In[41]:


#deine a list of correct and incorrect predictions and check the overall accuracy on the test data
acc_list=[]
def accuracy():
    correct =0
    for i in range(len(test_Y)):
            if (predicted[i] == test_Y[i]):
                acc_list.append('True')
            else:
                acc_list.append('False')
acc = accuracy()
#acc_list


# In[42]:


count =0 
for i in acc_list:
    if(i=="True"):
        count = count+1
    else:
        continue
print("Test accuracy for KNN classifier=",count/len(test_Y))


# # PCA

# In[43]:


#img[0].shape


# In[44]:


"""For PCA:
    1. Calculate the covariance matrix
    2. Calculate the Eigen vectors and values"""
 #Also, Note that img here is already mean centered.
    #append column vector using np.newaxis for co-variance matrix
v = np.asarray(img[0])[np.newaxis].T
for i in img:
     new_v = i[np.newaxis].T
     v = np.append(v, new_v, axis=1)
cov_matrix = np.cov(v) #using numpy.cov
display(cov_matrix)

print(v)


# In[45]:


#calculate eigen values and corresponding vectors from covariene matrix
eigen_vals, eigen_vecs = scipy.linalg.eigh(cov_matrix)


# In[46]:


# sort eigenvalue in decreasing order
idx = eigen_vals.argsort()[::-1]   
eigen_vals = eigen_vals[idx]
eigen_vecs = eigen_vecs[:,idx]


# In[47]:


#eigen_vecs


# In[48]:


scale=1
fig=plt.figure(figsize=(10*scale, 10*scale))
columns = 10
rows = 10
for i in range(1, 51):
    img_3 = convert_to_img(eigen_vecs[:,i]*1e4)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img_3)
fig.suptitle('Highest Eigenvectors(faces)', fontsize=14)
plt.show()


# # KNN with Dimensionality reduction using PCA

# In[49]:


#Select a range of eigen vectors by trial and error for good accuracy
new_eigen_vecs = eigen_vecs[:,5:10]


# In[50]:


#Projecting the datapoints on these calculated eigenvectors. this is obtained by np.dot b'w data and eigenvector
pca_train_X = []
pca_test_X = []
for i in range(len(train_X)):
    pca_train_X.append(np.dot(new_eigen_vecs.T,train_X[i]))

for i in range(len(test_X)):
    pca_test_X.append(np.dot(new_eigen_vecs.T,test_X[i]))


# In[51]:


#pca_train_X


# In[53]:


#Compute the accuracy
predicted_pca = []
for i in range(len(pca_test_X)):
    pca_neighbours = get_Neighbors(pca_train_X,train_Y,pca_test_X[i],1)
    predicted_pca.append(maximum_votes(pca_neighbours))

pca_acc_list=[]
def pca_accuracy():
    for i in range(len(test_Y)):
            if (predicted_pca[i] == test_Y[i]):
                pca_acc_list.append('True')
            else:
                pca_acc_list.append('False')
acc = pca_accuracy()
count =0 
for i in pca_acc_list:
    if(i=="True"):
        count = count+1
    else:
        continue
print("Test accuracy of KNN using PCA=",count/len(test_Y))

