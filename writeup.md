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
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
