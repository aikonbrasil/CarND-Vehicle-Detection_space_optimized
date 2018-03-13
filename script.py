import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
#import numpy as np
#import cv2
#import glob
import time
from sklearn.svm import LinearSVC
#from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
#from lesson_functions import *
from own_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
#%matplotlib inline
import matplotlib.image as mpimg
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
import time

# Loading training data provided by udacity
car_images = glob.glob('training_data/vehicles/**/*.png')
noncar_images = glob.glob('training_data/non-vehicles/**/*.png')
print(len(noncar_images), len(car_images))

# Getting features from dataset
color_space = 'YCrCb' #'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
# Bin spatial parameters
spatial_size = (16, 16) # Spatial binning dimensions
# Histogram parameters
hist_bins = 32    # Number of histogram bins
hist_range = (0, 256)
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [360, 700] # Min and max in y to search in slide_window()

car_features = extract_features(car_images)
notcar_features = extract_features(noncar_images)

# Training our classifier
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

# Split up data into randomized training and test sets
#rand_state = np.random.randint(0, 100)
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=rand_state)

# Fit a per-column scaler
#X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
#X_train = X_scaler.transform(X_train)
#X_test = X_scaler.transform(X_test)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

# SAVING with PICKLE
mydict = {"svc":svc, "scaler":X_scaler, "orient": orient, "pix_per_cell":pix_per_cell, "cell_per_block": cell_per_block, "spatial_size":spatial_size, "hist_bins": hist_bins}
output = open('info_trained_classifier.pkl','wb')
pickle.dump(mydict, output)
output.close()
#
# ystart = 350
# ystop = 700
# scale = 1
#
# # SLIDE WINDOW
# class parametros_video():
#     def __init__(self):
#         self.ystart = ystart
#         self.ystop = ystop
#         self.scale = scale
#         self.svc = svc
#         self.X_scaler = X_scaler
#         self.orient = orient
#         self.pix_per_cell = pix_per_cell
#         self.cell_per_block = cell_per_block
#         self.spatial_size = spatial_size
#         self.hist_bins = hist_bins
#
# from moviepy.editor import VideoFileClip
# from functools import reduce
#
# class HeatHistory():
#     def __init__(self):
#         self.history = []
#
# def processVideo(inputVideo, outputVideo, frames_to_remember=3, threshhold=1):
#     """
#     Process the video `inputVideo` to find the cars and saves the video to `outputVideo`.
#     """
#     history = HeatHistory()
#
#     def pipeline(img):
#         #return(img)
#         if True:
#             #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             parametros = parametros_video()
#             ystart = parametros.ystart
#             ystop = parametros.ystop
#             scale = parametros.scale
#             svc = parametros.svc
#             X_scaler = parametros.X_scaler
#             orient = parametros.orient
#             pix_per_cell = parametros.pix_per_cell
#             cell_per_block = parametros.cell_per_block
#             spatial_size = parametros.spatial_size
#             hist_bins = parametros.hist_bins
#
#             box_listt,window_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
#
#             print('dick')
#
#             heat = np.zeros_like(img[:,:,0]).astype(np.float)
#             heat = add_heat(heat,box_listt)
#
#             # Apply threshold to help remove false positives
#             heat = apply_threshold(heat,2)
#             #heatmap = apply_threshold(heat_history, 2)
#             labels = label(heat)
#
#             return draw_labeled_bboxes(np.copy(img), labels)
#
#     myclip = VideoFileClip(inputVideo)
#     output_video = myclip.fl_image(pipeline).subclip(1,2)
#     output_video.write_videofile(outputVideo, audio=False)
#
# processVideo('test_video.mp4', './video_output/project_video_teste.mp4', threshhold=2)
