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


