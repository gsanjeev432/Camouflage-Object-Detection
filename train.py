# Importing the necessary modules:
from skimage.feature import hog
from sklearn.externals import joblib
from skimage import feature
import numpy as np
import os
from PIL import Image # This will be used to read/modify images (can be done via OpenCV too)
from numpy import *
import warnings
warnings.filterwarnings("ignore")
from skimage import data

PERSON_WIDTH = 50
PERSON_HEIGHT = 100
leftop = [16, 16]
rightbottom = [16 + PERSON_WIDTH, 16 + PERSON_HEIGHT]

radius = 3
n_points = 8 * radius
METHOD = 'uniform'

def describe(image, radius, n_points,METHOD,eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, n_points,
			radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, n_points + 3),
			range=(0, n_points + 2))
 
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
 
		# return the histogram of Local Binary Patterns
		return hist

# define path to images:

pos_img_dir  = r"C:/Users/RAJIV/Downloads/Compressed/object-detector-master/object-detector/INRIAPerson/train_64x128_H96/pos/" # This is the path of our positive input dataset
# define the same for negatives
neg_img_dir = r"C:/Users/RAJIV/Downloads/Compressed/object-detector-master/object-detector/INRIAPerson/Train/neg/"

pos_img_files = os.listdir(pos_img_dir)
neg_img_files = os.listdir(neg_img_dir)

X = []
y = []
print('start loading ' + str(len(pos_img_files)) + ' positive files')
for pos_img_file in pos_img_files:
    pos_filepath = pos_img_dir + pos_img_file
    pos_img = data.imread(pos_filepath, as_grey=True)
    pos_roi = pos_img[leftop[1]:rightbottom[1], leftop[0]:rightbottom[0]]
    fd_hog = hog(pos_roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=False)
    fd_lbp = describe(pos_roi, radius, n_points,METHOD)
    fd = np.hstack([fd_lbp,fd_hog])
    X.append(fd)
    y.append(1)
    
print('start loading ' + str(len(neg_img_files)) + ' negative files')
for neg_img_file in neg_img_files:
    neg_filepath = neg_img_dir + neg_img_file
    neg_img = Image.open(neg_filepath)
    neg_img = neg_img.convert('L')
    neg_roi = neg_img.resize((50,100))
    fd_hog = hog(neg_roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=False)
    fd_lbp = describe(neg_roi, radius, n_points,METHOD)
    fd = np.hstack([fd_lbp,fd_hog])
    X.append(fd)
    y.append(0)
    
X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

from sklearn import svm

print('start learning SVM.')
lin_clf = svm.LinearSVC()
lin_clf.fit(X, y)
print('finish learning SVM.')
print(lin_clf.fit(X,y))
print(lin_clf.score(X,y))


##%% Save the Model
joblib.dump(lin_clf, 'modell_name.npy')