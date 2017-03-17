# Vehicle Detection Project
## Writeup
## Lucas Gago

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_nocars.png
[image2]: ./examples/orient.png
[image3]: ./examples/pixels.png
[image4]: ./examples/cell.png
[image5]: ./examples/color.png
[image6]: ./examples/final.png
[image7]: ./examples/grid.jpg
[image8]: ./examples/heat.png
[image8]: ./examples/label.png
[image01]: ./examples/simple.png
[video1]: ./output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.


I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` (left) and `non-vehicle` (right) classes:

<div style="text-align:center">

![alt text][image1]

<img src ="..." /></div>

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

I ploted the difference between orientation values:

<div style="text-align:center">

![alt text][image2]

<img src ="..." /></div>

Pixels per cell value:

<div style="text-align:center">

![alt text][image3]

<img src ="..." /></div>

And color spaces:

<div style="text-align:center">

![alt text][image5]

<img src ="..." /></div>



#### 2. Explain how you settled on your final choice of HOG parameters.

I tried several combinations of parameters and found that the best were:
```
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 7  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

```


![alt text][image6]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I made a loop using sklearn most common classfiers and testing its performance. I got the following results:

|Clasificator   	| Training  	|  Test 	|
|---	|---	|---	|
|  KNeighborsClassifier 	|  0.9896 	|   .9719	|
|SVC(kernel="linear", C=0.025)   	|  1.0000 	|  .9938 |
| SVC(gamma=2, C=1)  	| 1.0000  	|  0.5021 	|
| GaussianProcessClassifier  	| 1.0000  	|  0.5031 	|
| DecisionTreeClassifier  	| 0.9779  	|  0.9448 	|
| RandomForestClassifier  	| 0.9129  	| 0.8750  	|
| MLPClassifier  	|  1.0000 	|  0.9959 	|
| AdaBoostClassifier  	|  1.0000 	|  0.9875 	|
|  GaussianNB 	|   0.9646	|   0.9594	|
|QuadraticDiscriminantAnalysis| 1.0000|0.5802|


Also, an small NN architecture with keras give me good results, with a maximum of .994

Later, I optimize SVC and MLP( using `GridSearchCV` with selected parameters), getting the following results on the full dataset:

### SVC:

**Optimization**
Tuned model has a training accuracy score of 0.9932.

Tuned model has a testing accuracy score of 0.9917.

The best parameters are {'gamma': 5e-05, 'kernel': 'rbf', 'C': 2}

CPU times: user 49min 14s, sys: 0 ns, total: 49min 14s

Wall time: 49min 14s

**Full Dataset**

17.32 Seconds to train SVC...

Test Accuracy of SVC =  0.9958

21.62 Seconds to test accuracy...


### MLP
**Optimization**

Tuned model has a training accuracy score of 0.9940.

Tuned model has a testing accuracy score of 0.9885.

The best parameters are {'activation': 'logistic', 'solver': 'lbfgs', 'alpha': 1}

Wall time: 25min 35s

**Full Dataset**

22.87 Seconds to train SVC...

Test Accuracy of SVC =  0.9979

22.91 Seconds to test accuracy...


[Multilayer Perceptron Classifier](https://en.wikipedia.org/wiki/Multilayer_perceptron) is the best fit for this problem as we can see.



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

With the function `slide_window`, I get a list of all the windows I need, between some y limits (no cars in the sky). 

<div style="text-align:center">

![alt text][image7]

<img src ="..." /></div>

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I made a heatmap to avoid false positives. Here is an example:

<div style="text-align:center">

![alt text][image10]

<img src ="..." /></div>

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output.mp4). Most of the time it works right, but there is room for improvement.


<div style="text-align:center">

![alt text][video1]

<img src ="..." /></div>
#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. With the class `Processor` I accumulate positive windows and make a heatmap of 5 frames. This helps to get rid of false positives.

### Here are six frames and their corresponding heatmaps:

<div style="text-align:center">

![alt text][image8]

<img src ="..." /></div>


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Even though I managed to get a pretty good classifier, my system is really slow, not getting more than 1,5 fps. This is inadmissible for a real aplication, maybe looking into a full resolution image is not necesary, but it needs optimization. Filtering the ouptup with the classs `Video_Process` help me to get rid of false positives and make a smoother detection.

### Extras

Just for fun, I added a Haar Cascade Clasificator, and used ProcessorHaar to get rid of false positives, with non great results, but incredibly fast prossesing time.
