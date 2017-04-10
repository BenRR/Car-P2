# Traffic Sign Recognition
---
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: images/sample.png "Sample Image"
[image2]: images/stand_sample.png "Standardized Sample"
[image3]: images/train_hist.png "Training Set Distribution"
[image4]: images/valid_hist.png "Validation Set Distribution"
[image5]: images/test_hist.png "Test Set Distribution"
[image6]: images/web_test_images.png "Traffic Sign Images From Internet"
[image7]: images/web_test_top_k.png "Top 5 Predictions"
[image8]: images/featuremaps_visual.png "CNN Feature Map Visualization"
[image9]: images/test1.png "1st web image"
[image10]: images/test2.png "2nd web image"
[image11]: images/test3.png "3rd web image"
[image12]: images/test4.png "4th web image"
[image13]: images/test5.png "5th web image"
[image14]: images/prob_hist.png "prob distribution"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my project code ( [html](Traffic_Sign_Classifier.html) / [notebook](Traffic_Sign_Classifier.ipynb) )

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* Number of training set = 34799
* Number of validation set = 4410
* Number of testing set = 12630
* Image data shape = 32 * 32
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

I used matlibplot to show the distribution of training, validation and test sets:

![training distribution][image3] ![validation distribution][image4] ![test distribution][image5]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

I decided to use `per_image_standardization` method from tensorflow to standardize images.

Here is an example of a traffic sign image before and after standardization.

Before ![raw sample image][image1]
After  ![standardized image][image2]

I decided not to use grayscale because I would like to know how good the CNN performs with colorful images. And I did not generate more training data at the beginning because I wanted to know how well my model performs without any extra data. Since the result of my data made the fine line, I didn't come back to create more training images.

#### 2. Describe what your final model architecture looks like.

I used the LeNet from lecture as a base architecture and also take google's LeNet implementation as a reference when choosing the default hyperparameters.
My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 1 5x5     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling 1	 3x3     	| 2x2 stride, same padding, outputs 16x16x64 				|
| Normalization					|												|
| Convolution 2 5x5	    | 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Normalization					|												|
| Max pooling 2	 3x3     	| 2x2 stride, same padding, outputs 8x8x64 				|
| Flatten    	| outputs 4096				|
| Fully connected	1	| 4096x384, outputs 384        									|
| RELU					|												|
| Fully connected	2	| 384x192, outputs 192        									|
| RELU					|												|
| Fully connected	3	| 192x43, outputs 43 number of label classes     									|
| Softmax				| for cross-entropy loss        									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an `AdamOptimizer` from tensorflow and use `softmax_cross_entropy_with_logits` methods to calculate the loss also I add regularization l2 loss to avoid overfitting (more explanation in the approach section).

My final hyperparameters:
* EPOCHS = 25
* BATCH_SIZE = 400
* L2_PENALTY = 0.012
* learing_rate = 0.0008
* dropout keep_prob = 0.4

#### 4. Describe the approach taken for finding a solution.

I use only 25 EPOCHS because after that the accuracy did not improve much. The last 3 EPOCHS results look like

| EPOCH         		|    Training Accuracy	    | Validation Accuracy |
|:---------------------:|:---------------------:|:------------------------:|
| 23         		|    0.998    | 0.949 |
| 24         		|    0.999    | 0.967 |
| 25         		|    0.999    | 0.952 |

* Final training accuracy is 0.999
* Final validation accuracy is around 0.95 ~ 0.96
* Final test accuracy = 0.957

If an iterative approach was chosen:
* I started with LeNet from the lecture the I got training accuracy around 0.92
* Then I adjust the fully connected layers' nodes according to google's LeNet implementation and reduced the learning rate
* And training accuracy was really good, around 0.99 after 30 EPOCHS but validation accuracy was very low, around 0.75
* I believed that it was a clear sign of overfitting.
* So I made two changes. 1) add L2 regularization loss 2) introduce dropout keep_prob 0.5
* Then I saw a great improvement of validation accuracy to 0.90 which I believed still overfitting
* Then I tuned the following hyperparameters, increased the L2 regularization penalty and decreased dropout keep_prob from 0.5 to 0.4
* Finally the validation accuracy can reach above 0.95 then I increased the learning_rate a little to fast the training process

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![test images from web][image6]

I think the first, second and last images should be easy to classify because the signs are at the center of the pictures without much noise.
Third and forth images have some background colors might cause some difficulties.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.
Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Roundabout mandatory      		| Roundabout mandatory   									|
| Speed limit (60km/h)     			| No passing 										|
| Right-of-way at the next intersection					| Right-of-way at the next intersection											|
| Speed limit (20km/h)	      		| General caution					 				|
| Children crossing			| Children crossing      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of 95.7%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

![web image 1][image9]

For the first image, the model is confidently sure that this is a "Roundabout mandatory" (probability of 1.0), and the image is a roundabout. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Roundabout mandatory    									|

![web image 2][image10]

For the second image, the model is relatively sure that this is a "No passing" (probability of 0.82), unfortunately it is horribly wrong, the image is a Speed limit (60km/h). I suspect the horizontal metal bar in the picture confused the model. The top five soft max probabilities all contains horizontal bar shape block.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .82         			| No Passing    									|
| .069    				| Keep left 										|
| .043					| No vehicles											|
| .028	      			| No passing for vehicles over 3.5 metric tons				 				|
| .016				    | Double curve      							|

![web image 3][image11]

For the third image, the model is very sure that this is a "Right-of-way at the next intersection" (probability of 0.997), and the image is a right-of-way at the next intersection. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .997         			| Right-of-way at the next intersection    									|
| .001     				| Double curve										|
| .001					| Pedestrians											|

![web image 4][image12]

For the first image, the model thinks that this is a "General caution" (probability of 0.349), and the image is a Speed limit (20km/h) which is also in the top five predictions with probability 0.115. And the second prediction is "Speed limit 30km/h" with a quite high probability. The distribution of prediction probability is:

![prediction distribution][image14]

The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .349         			| General caution    									|
| .325     				| Speed limit (30km/h) 										|
| .115					| Speed limit (20km/h)											|
| .09	      			| End of speed limit (80km/h)					 				|
| .0052				    | End of all speed and passing limits      							|

![web image 1][image13]

For the last image, the model is quite sure that this is a "Children crossing" (probability of 0.998), and the image is a 0.998. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .998         			| Children crossing    									|
| .002     				| Right-of-way at the next intersection 										|

### Visualizing the Neural Network
Original image

![web image 1][image9]

Featuremaps of Conv layer 1

[image8]: images/featuremaps_visual.png "CNN Feature Map Visualization"

We can see that the curve of the circles of the roundabout also the three round arrows are all picked up by the featuremaps.
