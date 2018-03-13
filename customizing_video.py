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

# LOAGIN a pre trained svc model from a serialized files
dist_pickle = pickle.load( open("info_trained_classifier.pkl", "rb" ) )
print('Loaded info from pickle')
# get attributes of our svc object
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

ystart = 350
ystop = 700
scale = 1

# SLIDE WINDOW
class parametros_video():
    def __init__(self):
        self.ystart = ystart
        self.ystop = ystop
        self.scale = scale
        self.svc = svc
        self.X_scaler = X_scaler
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins

from moviepy.editor import VideoFileClip
from functools import reduce

class HeatHistory():
    def __init__(self):
        self.history = []

def processVideo(inputVideo, outputVideo, frames_to_remember=5, threshhold=1):
    """
    Process the video `inputVideo` to find the cars and saves the video to `outputVideo`.
    """
    history = HeatHistory()

    def pipeline(img):
        #return(img)
        if True:
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            parametros = parametros_video()
            ystart = parametros.ystart
            ystop = parametros.ystop
            scale = parametros.scale
            svc = parametros.svc
            X_scaler = parametros.X_scaler
            orient = parametros.orient
            pix_per_cell = parametros.pix_per_cell
            cell_per_block = parametros.cell_per_block
            spatial_size = parametros.spatial_size
            hist_bins = parametros.hist_bins

            #box_listt,window_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
            box_listt=find_cars_opt(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

            heat = np.zeros_like(img[:,:,0]).astype(np.float)
            heat = add_heat(heat,box_listt)

            if True:
                ## custome selection
                if len(history.history) >= frames_to_remember:
                    history.history = history.history[1:]
                    #print(type(history.history))
                    #print('#Rows = ',len(history.history[0]))
                    #print('#Cols = ',len(history.history))

                history.history.append(heat)
                heat_history = reduce(lambda h, acc: h + acc, history.history)/frames_to_remember
                heat = heat_history
                ## END OF CUSTOME selection

            # Apply threshold to help remove false positives
            heat = apply_threshold(heat,2)
            #heatmap = apply_threshold(heat_history, 2)
            labels = label(heat)

            return draw_labeled_bboxes(np.copy(img), labels)

    myclip = VideoFileClip(inputVideo)
    output_video = myclip.fl_image(pipeline).subclip(8,20)
    #output_video = myclip.fl_image(pipeline)
    output_video.write_videofile(outputVideo, audio=False)

processVideo('project_video.mp4', './video_output/project_video_final.mp4', threshhold=2)
