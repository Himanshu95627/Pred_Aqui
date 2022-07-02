# Startup’s Acquisition predictor
Internship Leader: Yasin Shah
Mentor: Austin
Group members: Dipesh chatrola, Himanshu Yadav, Veda Vyaas K


## Project Description

This project specifically focuses on finding an alternative method to predict a Startup’s acquisition status based on its financial statistics by using different machine learning techniques.


Financial information about the firms, including total funding, funding dates, total number of fundraising rounds, and headquarters location, is fed into the resulting algorithm. The system then makes a prediction on whether the startup has been closed down, acquired, is still running, or has achieved an IPO. Dealing with an unbalanced dataset where one class is overrepresented and under/oversampling cannot be used as a method to balance the data is the key challenge for this issue. We used an ensemble-based technique to address this.


Dataset:
In this project we have used the data collected from Crunchbase 2013 available on kaggle.

Each row of combined dataset contains information like: 'id', 'Unnamed: 0.1', 'entity_type', 'entity_id', 'parent_id', 'name', 'normalized_name', 'permalink', 'category_code', 'status', 'founded_at', 'closed_at', 'domain', 'homepage_url', 'twitter_username', 'logo_url', 'logo_width', 'logo_height', 'short_description', 'description', 'overview', 'tag_list', 'country_code', 'state_code', 'city', 'region', 'first_investment_at', 'last_investment_at', 'investment_rounds', 'invested_companies', 'first_funding_at', 'last_funding_at', 'funding_rounds', 'funding_total_usd', 'first_milestone_at', 'last_milestone_at', 'milestones', 'relationships', 'created_by', 'created_at', 'updated_at', 'lat', 'lng', 'ROI'.

## Data pre-processing

## A. Data Cleaning
1. Delete irrelevant & redundant information
2. Remove noise or unreliable data (missing values and outliers)
    
### 1. Delete irrelevant and redundant information
* Delete 'region','city','state_code' as they provide too much of granularity.
* Delete 'id', 'Unnamed: 0.1', 'entity_type', 'entity_id', 'parent_id', 'created_by',       'created_at', 'updated_at' as they are redundant.
* Delete 'domain', 'homepage_url', 'twitter_username', 'logo_url', 'logo_width', 'logo_height', 'short_description', 'description', 'overview','tag_list', 'name', 'normalized_name', 'permalink','invested_companies' as they are irrelevant features.
* Delete duplicate values if any.
* Delete those which has more than 98% of null values. As we mention before there were some columns which had more than 95% of null value so, if we fill the missing values it will give us the wrong accuracy.

### 2. Remove noise or unreliable data
* Delete instances with missing values for ‘status’, ‘country_code’, ‘category_code’ and ‘founded_at’.
* Delete outliers for ‘funding_total_usd’ and ‘funding_rounds’.
* Delete contradictory (mutually opposed or inconsistent data).

## Feature Enginerring
* Considered the suitable features for the model based upon the combined result of model performance and exploratory data analysis.
* Encoded features like `Country_code`, and `Catogery_code`.
* Created additional features like `isClosed` and `active_days`.

The final set.
![](https://i.imgur.com/6Red09z.png)
![](https://i.imgur.com/m2LU03s.png)

![](https://i.imgur.com/qMcGHOd.png)


## How we handle Data Imbalalance?
Imbalalance data can sometime affect our models performance and can bring biasness in our models performance. For e.g. In case of our data if we didn't handle the imbalance data then we might end up building a bias model with a lot of false positive value. That's quiet a problem. So we need to handle the imbalance dataset using

1. Undersampling of majority class label.
2. Oversampling of minority class label
3. Using SMOTE Technique which adds synthetic datapoints to our minority class label in order to balance the number of data points in each classes.

- Outliers
- Relationship between Independent and dependent feature


## Solution Appraoch
Since we have a multiclass imbalanced classification problem, we decided to break it down into subproblems. To do this, we first classified the startup's running state based on the "isClosed" column, and then we used the appropriate classification algorithms to predict the actual class based on the output of this classification.

![](https://i.imgur.com/z1aq6gn.png)


### Description of the first level Model

We used Logistic Regression as the first level classifier algorithm to train our dataset. After doing the necessary EDA and DataPreprocessing we splitted our dataset into train and test set. After that we scaled the training and validation data using min_max_scaler. After training the model we tested the model on our validation dataset and we achieve the accuracy nearly 100%.

![](https://i.imgur.com/1r2cxfC.png)

### Description of the second level classifiers
### RUNNING(1):
Since we had two subproblems to be solve at level-2 therefore we have used various machine learning algorithms on filtered data.

We have tried  different models i.e., Logistic Regression, KNN, Bagging Classifier, Random Forest and XGBoost. And from that we select **Bagging Classifier** as our final model for running_startup. As rest of the models are predicting bias towards operating only so we decided to go for Bagging Classifier.

As the dataset is totally imbalanced you can see below:
> ![](https://i.imgur.com/wGmmI5a.png)

After doing the necessary EDA and DataPreprocessing we splitted our dataset into train and test set. 
```
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)

print('X_train dimension:', X_train.shape)
print('X_test dimension:', X_test.shape)
print('y_train dimension:', y_train.shape)
print('y_train dimension:', y_test.shape)
```
> X_train dimension: (35627, 38)
> X_test dimension: (15270, 38)
> y_train dimension: (35627, 1)
> y_train dimension: (15270, 1)

After that we scaled the training and validation data using **StandardScalar**.
```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
```

We used the concept of **SMOTETomek** to increase the number of minority class.
```
os=SMOTETomek(0.70)
X_train_ns,y_train_ns=os.fit_resample(X_train,y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_ns)))
print("The number of classes before fit {}".format(y_train.value_counts()))
print("The number of classes after fit {}".format(y_train_ns.value_counts()))
```

> The number of classes before fit Counter({'status': 1})
> The number of classes after fit Counter({'status': 1})
> The number of classes before fit status   
> operating    35335
> ipo            292
> dtype: int64
> The number of classes after fit status   
> operating    35327
> ipo          24726
> dtype: int64

After applying Bagging Classifier with Decision Tree Model we got 99.99% training accuracy and 98.50% as testing accuracy


```
bag_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators =100,
    oob_score=True,
    random_state=0,
    max_samples=0.8
)

bag_model.fit(X_train_ns,y_train_ns)
```
We used Confusion Matrix and Classification Report to analyze the model’s performance:

```
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
```
![](https://i.imgur.com/nLYXJHs.png)

```
print(classification_report(y_test, y_pred))
```
> ![](https://i.imgur.com/8WU9ydK.png)


### CLOSED(0):
The **closed** category in the **isclosed** column corresponding to the status consists of two subclasses(Acquired and closed)

![](https://i.imgur.com/K9COhHp.png)



We can clearly see that the closed subclass has more count than the acquired subclass , so balancing of the classes needs to be done .

Before that the data needs to Scaled , for that purpose we use **StandardScaler** , which is used to resize the distribution of values so that the mean of the observed values is 0 and the standard deviation is 1.
![](https://i.imgur.com/SizG8IX.png)

Then as done before we split the dataset into train and test split's
```
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)
```
do SMOTE analysis to get the class distribution as follows

![](https://i.imgur.com/b4aVt3m.png)

After performing SMOTE analysis we use **BaggingClassifier** as compared to other classification algorithms such as XGBoost,SVC...etc as it gives us a test accuracy of 94.4%.Better than tthe other models

Finally from the confusion matrix we come to know about the classification of the TP's,TN's,FP's and FN's.
```
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
```
![](https://i.imgur.com/svqiJpU.png)

And the accuracy of the model came around 94.4%.



## Deployment
You can access our Django based web-app by following this link [PredAqui](https://predaqui.herokuapp.com/).

Django: Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design. Built by experienced developers, it takes care of much of the hassle of web development, so you can focus on writing your app without needing to reinvent the wheel. It’s free and open source.

Later we used Heroku as our platform for deployment. Heroku is a cloud platform service (PaaS) supporting several programming languages. One of the first cloud platforms, Heroku, has been developing since June 2007, when it only supported the Ruby programming language. It now supports Java, Node.js, Scala, Clojure, Python, PHP, and Go. For this reason, Heroku is said to be a polyglot platform as it has features for a developer to build, run and similarly scale applications across most languages.
