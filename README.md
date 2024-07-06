# MarketingModels
AI ML DL models related to marketing - product attribution, churn, marketing mix

Product Attribution

Product attribution is the process of quantifying the contribution of different product attributes to a business outcome like sales, customer engagement, demand prediction. They are used to assign credit for conversions across multiple touchpoints in a customer's lifecycle during sales & marketing, e-commerce, demand forecasting, other business outcomes.

Machine learning/ AI models can be used to determine which product features (price, brand, design, reliability, others) drive customer sales and engagement through different channels across multiple touchpoints in a customer's lifecycle.
It is effectively identifying feature importance of product attributes across the customer lifecycle used to predict a target variable like sales. Some examples are
•	Sales and marketing: clicks & views in email, social media, tv and print ads
•	E-commerce: search ads click through rates, purchase, cart abandonment
•	Targeted marketing: personal recommendations using customer segmentation, socio economic profile, product attributes
•	Demand forecasting: supply chain optimization using inventory and sales data

Various machine learning / AI models can be used for quantifying product attributes:

Heuristic model attribution
1.	Last touch: 100% attribution to last conversion touchpoint
2.	First touch: 100% attribution to first conversion touchpoint
3.	Time decay: exponential decay rate to give higher attribution to most recent touchpoints
4.	Distribution model: attribution to all touchpoints distribution using allocation %

ML AI models
Most AI ML models can be used for product attribution across multiple touchpoints depending on the type of analysis required. 

Classical ML models
1.	Regression: Predict future sales, product attributes & multiple touchpoints importance using regression models like random forest, xgboost, regularization while preventing overfitting
2.	Classification: KNN, Logistic regression, to identify attributes and channels that contribute to sales, engagement
3.	NLP: Sentiment analysis, topic model from feedback text, call center transcript, social media views used to understand likes, frustration, product features, demand
4.	Clustering: KMeans segmentation to identify customer groups for targeted marketing
5.	Time Series forecasting: ARIMA, other models for demand inventory optimization using past sales data allowing for trend, seasonality, regression, autocorrelation
6.	Recommendation systems: Collaborative filtering of product attributes to make recommendations to customers. Cosine similarity, matrix factorization, autoencoders
a.	User based: Using profile, preferences of similar customers
b.	Item based: Using customers past purchase products profile
c.	Content based: Using customers past purchase product attributes
7.	Optimization models: Using integer, MIP, Non-linear, heuristic optimization models to efficiently allocate resources across the supply chain to most important product attributes for customer sales and engagement

AI, Deep Learning, Gen AI
1.	Transformers: BERT, GPT fine tuned models to analyze text from all customer touchpoints and predict behavior across the lifecycle
2.	Deep Learning: Custom multi-layer neural network models for predicting probability distribution of product attributes across customer profiles 
3.	Semantic search: enhance search using query expansion, understanding intent to make product recommendations, entity recognition, sentiment analysis
4.	AI chatbot: Provide customer support, recommendations using Gen AI models
5.	RNN/LSTM: Predict future sales by modelling past sequential data to learn long term non-linear dependencies in customer touchpoint lifecycle using neural nets
6.	CNN/ViT: Image analysis, object detection, tagging, categorizing using CV algorithms to improve product recommendations and identify product attributes improving engagement
7.	Multi-task learning: Combine input features (customer profile, product attributes, sales) in input and shared neural network layers to create task specific layers for business objectives like sales to make final prediction, using shared learned behavior

Data Science Process
All of the above models can be applied independently or stacked in varying combinations to get the best fit model for product attribution using the standard data science process:

1.	Describe the business problem
2.	Perform Exploratory Data Analysis
3.	Provide possible solutions and scenarios
4.	Algorithm (model) Selection
5.	Perform Data Wrangling (transformation, preprocessing)
6.	Split sample data into Dependent, Training, Validation and Test sets
7.	Standardizing (Scaling) the data
8.	Model Building, Evaluation & Selection 
9.	Data Visualization
10.	Deploy the final model to cloud 
11.	Update data, Retrain and Redeploy the model

CNN model to extract product attributes from image

Objective: Show a basic architecture and implementation of a CV (computer vision) CNN model to extract product attributes by using object detection, by training custom product image data on large pre-trained CNN models which can then be improved on by further tuning

1.	This initial model shows how to use CNN models trained on public datasets to perform object detection, tagging, categorizing to identify product attributes for improving customer engagement and product recommendations for sales
2.	The tags or labels are effectively the product attributes
3.	Model can be trained on custom data, on top of the pre-trained model
4.	Reinforcement Learning can then be used by freezing the good model layers and re-trained on new image data in an iterative improvement loop

Model Used: Used a pre-trained Faster R-CNN model with a ResNet-50 backbone and trained it on COCO dataset. There are many other models that can be used (YOLO, SSD, etc)

Reason: Both are popular and widely used models and datasets, with COCO having a good collection of image object labels which can be used for tagging new images. 

Basic model terms:
1.	R-CNN: Regions with Convolutional Neural Networks. It uses a region proposal algorithm to generate 2000 region proposals per image, which are then classified by a CNN
2.	Faster R-CNN: improves on R-CNN by sharing convolutional features across region proposals, reducing computation time
3.	ResNet-50: 50-layer deep residual network which has 50 layers that perform convolution, batch normalization, and activation operations
4.	Each layer itself has Input Layer, Convolutional Layers, Residual Blocks, Pooling Layers, Fully Connected Layers, Output Layer. This is a typical CNN model architecture
5.	COCO (Common Objects in Context): large-scale object detection, segmentation, and captioning dataset. It contains over 200,000 labeled images with 80 common object categories such as person, car, chair, etc. It has separate, training, validation and test sets

Overall model training/ Code overview:
1.	Using 2 RTX4090 GPU’s with 48 GB RAM each. Training takes ~ 20 mins
2.	Trained for simple 5 epochs on 2000 samples (out of ~ 118,000)
o	All model parameters are in the code
3.	Used small samples from the training set for both training and test, with distinct independent ranges to avoid data leakage / mixing data
o	This was due to test set data files giving lot of errors like size differences, etc, need to fix it before using
4.	Training loss reduces from 0.72 to 0.46. So model is good and can be improved
5.	Test Loss is 0.92, high, can be improved by using larger samples and more epochs
6.	Evaluating a random sample image from test data shows decent accuracy in identifying the object labels from the bounding boxes. Can be improved by training
7.	The object labels are effectively the product attributes which can be written to any database for downstream operations to build customer sales or engagement related models, once customer interaction lifecycle data is attached to image product attributes

Improvements:
1.	The model is basic and can be improved in various ways – different pre-trained models, custom dataset, more data, hyperparameter tuning, etc
2.	If we train and test on entire dataset, results will definitely improve. Needs overnight run
3.	Need to implement DataParallel functionality to ensure pytorch is using all GPU’s for training (ResNet-50 model allows this) to see if it makes any difference in speed

Sample Image with bounding box and labels(attributes):
