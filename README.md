1.	Problem Discussion
In a serious car accident, I focus on the things that matter the most: family, friends, and other loved ones. At that point in time pushing paper or opening a claim with insurance agent is the last thing I want. Our aim to develop a project for Allstate, a personal insurer in the United States to predict the cost and severity of claims and eventually improve the claims service to ensure a worry-free customer experience (source: https://www.kaggle.com/c/allstate-claims-severity). The data for this project have been downloaded from Kaggle. The final submission will be based on following:

	Mean Absolute Error(MAE) between the predicted loss and the actual loss
	Predicting the cost and severity of claims
	For every id in the test set, predicting the loss value
	Improve claims service

2.	Significance
I aspire to demonstrate insight into better ways to predict claims severity for the chance to be part of Allstate’s efforts to ensure a worry-free customer experience. The goal is to predict the loss based on the severity of the claims. By doing this in my opinion will enhance overall customer service which will be beneficial for the company as well as the claimant plus help the insurance company. The predictive model that I developed can help in 2 ways: -
	Recognition of potentially fraudulent claims
	Identification of potentially high value losses
	Effective way to root out claim volatility
	Preventable causes - such as inefficient processing, human error, outdated operational procedures and fraud.
	Early identification of claims with the potential for high leakage, thereby allowing for proactive management of the claim.
	Timely allocation of resources.
	Reserving /settlement values

3.	Exploratory Analysis
3.1	Data Understanding
The datasets I used in our project came from an on-going Kaggle(source: https://www.kaggle.com/c/allstate-claims-severity/data) competition. Each row in this dataset represents an insurance claim. 

train.csv - the training set
test.csv - the test set. You must predict the loss value for the ids in this file.
sample_submission.csv - a sample submission file in the correct format


3.2	Data Preparation and Feature identification
I should predict the value for the ‘loss’ column. Variables prefaced with ‘cat’ are categorical, while those prefaced with ‘cont’ are continuous. All ids are of 116 categorical features, 14 continuous features Loss (label to predict).


3.3	Missing Values
Considering the datasets this shows that the dataset is complete and there is no need of doing to clean it from empty entries’. there is no Null value of feature shape, texture and margin in the given dataset.


4.	Type of Models

I used both R and Python to generate the models. However, R seems to be an easy choice where I could do the analysis in a quick time. To train the data I have used 3 hidden layers Deep Learning algorithms with each of 1280 nodes and an epoch of 1000 using the h2o package on a subset of data which lasted for longer than 80 minutes.

Apart from 3 hidden layers Deep Learning algorithms using the h2o package, I also tried h20.Gradient Boosting Machine (GBM) algorithms and h2o.randomForest algorithms in R and Gradient Boosting Regression (GB) model in python. However, the best accuracy I got is with 3 hidden layers Deep Learning algorithms with each of 1280 nodes with leadership Board score of 1114.35807 which was better than the other three models.


I have produced 4 different output files for the loss values to show how predicting a loss value correctly can enhance overall claims experience for the customer as well as the Insurance company. These output files are produced using 4 different models: -

	h20 Deep Learning algorithm using R
	h20 GBM algorithm using R
	h20 Random Forest algorithm using R
	Gradient Boosting Regression algorithm using python

4.1	h2o Deep Learning using R
Deep learning tools in R is still relatively rare now when compared to other popular algorithms like Random Forest and Support Vector Machines. Before the discovery of H2O, my deep learning coding experience was mostly in SVM. Recently, I have started using h2o deep learning in R.(source: https://www.r-bloggers.com/things-to-try-after-user-part-1-deep-learning-with-h2o/).Deep Learning has cracked the code for training stability and generalization and scales on big data. It is often the algorithm of choice for highest predictive accuracy, as deep learning algorithms performs quite well in several diverse problems.
4.2	h2o Gradient Boosting Machine (GBM) using R

Gradient Boosting Machine builds gradient boosted classification trees and gradient boosted regression trees on a parsed data set and also it is one of the fast processed model ( which I noticed during my trails in AllState project)

4.3	h2o Random Forest using R

Random Forest (RF) is a powerful classification tool. When given a set of data, RF generates a forest of classification trees, rather than a single classification tree. Each of these trees generates a classification for a given set of attributes. The classification from each H2O tree can be thought of as a vote; the most votes determines the classification which could give better results provided I using mode trees.

4.4	Gradient Boosting Regression using Python
Gradient Boosted Regression or shorter Gradient Boosting is a flexible non-parametric statistical learning technique for classification and regression. Gradient boosting Regression is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.


5.	Formulation / Libraries

I have used several libraries to analyze and explore the data which are as follow:

5.1	 H2O cluster
H2O is fast, scalable, open-source machine learning and deep learning for Smarter Applications. Advanced algorithms, like Deep Learning, Boosting, and Bagging Ensembles are readily available for application designers to build smarter applications through elegant API’s. H2O implements almost all common machine learning algorithms, such as generalized linear modeling (linear regression, logistic regression, etc.), Naïve Bayes, principal components analysis, time series, k-means clustering, and others. H2O also implements best-in-class algorithms such as Random Forest, Gradient Boosting, and Deep Learning at scale. Customers can build thousands of models and compare them to get the best prediction results

5.2	cv2
It is used to extract different features from the Images, like area, contour, perimeter, diameter, boundaries.

5.3	Numpy
NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.

5.4	Pandas
pandas is a software library written for the Python programming language for data manipulation and analysis. It offers data structures and operations for manipulating numerical tables and time series. pandas are free software released under the three-clause BSD license.

5.5	Label Encoder
Label encoder is used to encode the values between 0 and n depends upon the different distinct values.

5.6	StandardScaler
Standardize features by removing the mean and scaling to unit variance. Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on later data using the transform method.

6.	Model Performance

I have Enhanced the model performance from (Kaggle competition score) 1.80485 to 0.02139 by adding several features and by tuning the parameters. 


6.1	h2o Deep Learning using R
Deep learning tools in R is still relatively rare at the moment when compared to other popular algorithms like Random Forest and Support Vector Machines. Before the discovery of H2O, my deep learning coding experience was mostly in SVM. Recently, I have started using h2o deep learning in R. (source: https://www.r-bloggers.com/things-to-try-after-user-part-1-deep-learning-with-h2o/).To train the data I have used 3 hidden layers Deep Learning algorithms with each of 1280 nodes and an epoch of 1000 using the h2o package on a subset of data which lasted for longer than 80 minutes.
Apart from 3 hidden layers Deep Learning algorithms using the h2o package, I tried using more hidden layers but it didn’t show much improvement in the performance rather the processing time was more in comparison to the old tuning parameters. However the best accuracy I got is with 3 hidden layers Deep Learning algorithms with each of 1280 nodes with leadership Board score of 1114.35807.


6.2	h2o GBM using R
Also, I can tune our GBM more and surely get better performance. The GBM will converge a little slower for optimal accuracy, so if I were to relax our runtime requirements a little bit, I could balance the learn rate and number of trees used. Using `h20 Gradient Boosting Machine` algorithms the best accuracy I got is with leadership Board score of 1158.8236. When I have used GBM model I have taken various parameters to tune the performance, however after tuning the parameters the score is limited and there are very less chances to improve. So, that’s why I have dropped GBM. By further enhancing the parameter I have faced the issue of overfitting the models which cause even more depletion in performance.


6.3	H2o Random Forest using R
I could further experiment with deeper trees or a higher percentage of columns used (mtries). The general guidance is to lower the number to increase generalization (avoid overfitting), increase to better fit the distribution. Though usually unnecessary, if a problem has a very important categorical predictor, this can improve performance. In this Allstate project where fine-grain accuracy is beneficial, it is common to set the learn rate to a very small number, such as 0.01 or less, and add trees to match. Use of early stopping is very powerful to allow the setting of a low learning rate and then building as many trees as needed until the desired convergence is met. Using `h2o.randomForest algorithms` best accuracy I got is with leadership Board score of 1258.369.


6.4	Gradient Boosting Regression using Python
Gradient Boosting for regression builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative gradient of the given loss function. As the part of the parameter tuning used different parameters, in tuning parameter I have used n_estimators as 100, learning rate as 0.1 , maximum depth as 1 and loss as ls and fitted the model based on the same parameters . generated the predicted values using the same parameter. Using ` Gradient Boosting regression in Python` best accuracy I got is with leadership Board score of 1123.236.

7.	Limitations

8.1	Process Time
Each model took a lot of time to process the execution which has certainly became a drawback because of which I couldn’t make more hit and trails to the exiting model’s.

7.2	Complexity
Most of the data was categorical which limits the usage of the models however I have changed into numeric values but that didn’t show much improvement in the performance and model accuracy. 

7.3	Kaggle Limitations
I have limited upload of the submission file as due to which much more experiments are limited.

8.	Learning

	I have got the more exposure to various algorithms and classifiers, I have learned how to tune parameters.
	Understood how the GBM algorithm and Random Forest algorithm(model) works. Now I feel that the GBM is close to the initial random forest models in their performance. However, I used a default random forest. 
	Random forest's primary strength is how well it runs with standard parameters. And while there are only a few parameters to tune, I can experiment with those to see if it will make a difference. The main parameters to tune are the tree depth. 
	While H2O Deep Learning has many parameters, it was designed to be just as easy to use as the other supervised training methods in H2O. 
	Also one remarkable observation was that with the early stopping, automatic data standardization and handling of categorical variables and missing values and adaptive learning rates (per weight) reduce the number of parameters to specify. 
	And also it's just the number and sizes of hidden layers, the number of epochs and the activation function can be tuned to get better accuracy.


