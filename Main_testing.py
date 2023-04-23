import keras
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob





from keras.preprocessing import image                  
from tqdm import tqdm

# Note: modified these two functions, so that we can later also read the inception tensors which 
# have a different format 
def path_to_tensor(img_path, width=224, height=224):
    # loads RGB image as PIL.Image.Image type
    print(img_path)
    img = image.load_img(img_path, target_size=(width, height))
    # convert PIL.Image.Image type to 3D tensor with shape (width, heigth, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, width, height, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_paths, width, height)]
    return np.vstack(list_of_tensors)



# callback to show the total time taken during training and for each epoch
class EpochTimer(keras.callbacks.Callback):
    train_start = 0
    train_end = 0
    epoch_start = 0
    epoch_end = 0
    
    def get_time(self):
        return timeit.default_timer()

    def on_train_begin(self, logs={}):
        self.train_start = self.get_time()
 
    def on_train_end(self, logs={}):
        self.train_end = self.get_time()
        print('Training took {} seconds'.format(self.train_end - self.train_start))
 
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start = self.get_time()
 
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_end = self.get_time()
        print('Epoch {} took {} seconds'.format(epoch, self.epoch_end - self.epoch_start))

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

from tkinter import filedialog
filename = filedialog.askopenfilename(title='open')
test_tensors = paths_to_tensor(filename)/255


from keras.models import load_model


model = load_model('trained_modelRCNN.h5')

pred=model.predict(test_tensors)

print(pred)

if np.argmax(pred)==0:
    print('Given Input images is First degree of skin burn')
if np.argmax(pred)==1:
    print('Given Input images is First Regular/Normal skin')
if np.argmax(pred)==2:
    print('Given Input images is Second degree of skin burn')
if np.argmax(pred)==3:
    print('Given Input images is Third degree of skin burn')


######

# Python program to transform an image using 
# threshold. 
import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
  
# Image operation using thresholding 
img = cv2.imread(filename) 

cv2.imshow('Original Image',img)

# convert to RGB
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)
# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# number of clusters (K)
k = 5
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()


# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]

# reshape back to the original image dimension
segmented_image = segmented_image.reshape(image.shape)
# show the image
plt.imshow(segmented_image)
plt.show()
a1=0
a2=0
a3=0
a4=0
a5=0
for i in range(len(labels)):

    if labels[i]==0:
        a1=a1+1
    if labels[i]==1:
        a2=a2+1
    if labels[i]==2:
        a3=a3+1
    if labels[i]==3:
        a4=a4+1
    if labels[i]==4:
        a5=a5+1




xc=min([a1,a2,a3,a4,a5])
print('Afferected Area is ='+str(xc)+ '  Pixels')
print('Afferected Area in Percentage ='+str((xc/len(labels))*100)+' %')




