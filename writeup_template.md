## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/cars_classes_0.png
[image2]: ./writeup_images/generic_channel_HUG.png
[image3]: ./writeup_images/script_training_result.png
[image4]: ./writeup_images/sliding_window_0.png
[image5]: ./writeup_images/sliding_windows_1.png
[image6]: ./writeup_images/sliding_windows_2.png
[image7]: ./writeup_images/image_intro_video.png
[video1]: ./video_output/project_video_final.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.    

You're reading it!. It is important to mention that in this Writeup I am using information of the jupyter notebook called `project_solution.ipynb`, I am also using code script used in the files `own_functions.py`, `script.py`, and `customizing_video.py`. Basically the jupyter notebook contains the same code used in python code files. The main reason to use `script.py` was to perform the specific tasks of features selection and training of the classifier, the output of this script is a file called `info_trained_classifier.pkl` that contains the `svc` trained classifier that is opened by the `customizing_video.py` file code, this code was used exclusively to improve the methods to get the final output video. 

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook called `project_solution.ipynb` inside the section called `Histogram of Oriented Gradients `(or the customized script of HOG function between lines 54 through 55 of the file called `script.py`). It is important to mention that I have re-used the Udacity code with the function `extract_features(DATA_SET) `.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using an generic channel and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. In this case I used the function `get_hog_features()` that is defined in the file library called `own_functions.py` in the line 65:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried many parameter combinations. In this stage, I used also Udacity code. However parameter selection was done using a custom function called `tweak_function_hug_extracting_features()`. With this function I create man combinations with different values for the next parameters: `colorspace`, `orient`, `pix_per_cell`, `cell_per_block`, `hog_channel`. For this specific case I fixed the `cells_per_block=(2,2)` and `pix_per_cell=(8,8)` and run a loop to check configurations for different `colorspace` parameters and `orient` that was checked into a range between `9` and `13`.

The criteria was to evaluate the accuracy of classifier for each scenario described before and choose the best one (the one with best accuracy). The best configuration of parameters was: `color_space='YCrCb', spatial_size=(16, 16),hist_bins=32, orient=8,pix_per_cell=8, cell_per_block=2, hog_channel="ALL",hist_range=(0, 256)`. This parameters values were considered in the optimized code in the file `own_function.py` since the line code number 8.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the 7th code cell of the IPython notebook called `project_solution.ipynb` there is a section called "Training the classifier with dataset of cars and non cars groups". In this section I defined the definitive extracting features parameters of function called `extract_features()` (function defined in line 8 of library `own_function.py`). In the next cell (labeled as training classifier) I created an array stack of feature vectors (car and non-cars), I defined the labels vector, I scaled and normalized using `standardScaler()` function. After that I split the dataset in randomized training sets (100 classes of datasets for each train session) and also I divided the dataset in dataset for training (X_train, X_test, y_train, and y_test).

The next important step was related to classify train function using SVC. I used the code shared by Udacity and performed the training and finally I finished this step with the accuracy evaluation. Check the accuracy rate using the previous parameters as the final result of the classify train (Test `Accuracy of SVC = 0.9938`).

![alt text][image3]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search using sub-sample technique following the code provided by Udacity. So, in the IPython notebook called ` project_solution.ipynb` in the cell 9 there is a section dedicated to slide window. In this cell I defined a helper function called `getting_windows_info()`, this function was done based on the udacity code, specifically I used the functions `slide_window()` and `search_windows()` that were included in the script of the library called `own_functions.py` since the line 81.

![alt text][image4]

The same sliding window search was optimized with the function `find_cars()`  code provided in the Object Detection lesson (check code line 223 in `own_function.py` library function ). In this step I taken advantage of the tips provided by udacity to use sub-sampling window search. Check the result of this step.

![alt text][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I performed the search with the same code provided by Udacity in the lesson. However I changed an important parameter that improve so much the result, it was related to the line `260` in `own_functions.py` the parameter `cells_per_step=1`. As suggested in the code I performed the search using using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. All these configurations provided a nice result.  Here are some example images:

![alt text][image6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)


Here's a ![Link to my video result][video1] or in case it is not possible to play in the browser, you can check it in Youtube ![https://youtu.be/jV1pgFoNyVo]. As you can check, there are a minimal false positives in the shadow region of the video. But in general, the car detection was well performed.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As previous projects, I imported the moviepy.editor function (check line 63 in `customized_video.py` file), with this function I recover each video frame and processed it with the function `output_video = myclip.fl_image(pipeline)` (check line 119 in `customized_video.py` file). The `pipeline` function was defined since line 76 in the same `customized_video.py` file. Here, I recover parameters used in previous steps (Features extraction and classifier parameters) using the class called `parametros` and used an optimized `find_cars()` function that was a new name (`find_cars_opt()`), this code was optimized with the objective to reduce the time processing of the video final output.

In the same `pipeline` function I inserted a simple algorithm to consider historical information and applied a mean of the las 5 frames in order to obtain a smooth visualization when blue boxes were plotted over the detected cars.

I also applied the suggested method to avoid false positives using the function `add_heat()` defined in `own_functions.py` library and used in the line 96 in `customized_video.py` file. Using the output of `add_heat()` I also applied the `apply_threshold()` using a threshold equal to `2` with a great results as you can check in the final output video. Finally I taked advantages of the label functions as suggested by Udacity in the lessons.


Here's an example result showing the heatmap from a series of frames of video:


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all five frames,the fist column is showing the `find_cars_opt()` function, the second column is showing the final output of `draw_labeled_bboxes()` (check line 116 in `customizing_video.py` file) and finally the column 3 showing the heat_map used to avoid false positives:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As I commented before, It looks that the classifier is not 100% accurate. That is why that some false positives were shown even using a false positive technique. Maybe a good option is to equalize other feature extraction that I do not evaluated. I mean, `cells_per_block=(2,2)` and `pix_per_cell=(8,8)` could be evaluated to other values.

Other important issue is the time processing of the `customizing_video.py` file code. Even that I saved the SVN classifier in pickle file, The processing video spent so much time.
