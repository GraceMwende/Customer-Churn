# Customer-Churn
# SyrialTel Customer Churn

## Problem Statement: Predicting Customer Churn for SyriaTel
`Customer Churn` refers to the phenomenon where customers stop using a company's products or services. In telecommunications industry, churn occurs when a subscriber cancels their service, switches to a competitor, or stops engaging with the company altogether

For Syrialtel, a telcom provider, high churn rates lead to significant revenue losses, increased customer acquisition costs, and a weakened market position.Retaining existing customers is generally more cost-effective than acquiring new ones, making churn prediction a critical business priority.

## Disadvantages of Customer Churn:
1. Revenue loss - Losing customers reduces recurring revenue, impacting overall profitability
2. Higher Acquisition Costs - Acquiring new customers is often more expensive than retaining existiong ones.
3. Reputational Damage - High churn ratess may inidcate poor service quality,leading to negative word-of-mouth
4. Reduced Customer Lifetime Value (CLV) - Frequent customer exits lower the long-tern revenue a company can generate from each user
5. Operational Inefficiencies - constantly replacing lost customers requires continous marketing and sales efforts, increase costs

## Objective
The goal is to build a predictive model that identifies customers who are likely to churn in the near future. By analyzing patterns in customer behavior, the company can implement targeted retention strategies, such as personalized offers, improved customer support, or proactive engagement, to reduced churn and enhance customer loyalty

# Summary

## Dataset Overview
- **3333 rows and 21 columns** with features relating to customer behavior, call usage, subscription plans and churn status

## Key features
- **Categorical**: `state,international plan, voice mail plan`
- **Numerical**: `total day minutes,total eve minutes,total night minutes,customer service calls,account length`
- **Target Variable**: `churn`(binary:True=Churned,False=Retained)
## Data Cleaning
- The dataset contains **no missing values** and **no duplicates**
- Dropped irrelevant features: `Phone number` and `area code`(no impact on churn)
- Dropped highly correlated features to avoid multicollinearity(`total day charge, total eve charge, total night charge, total intl charge`)

## Data Preparation
- One-Hot Encoding was applied to categorical features
- Used Standard Scaler for numerical features to normalize data
- Applied log transformation to skewed numerical features(`total intl calls,customer service calls`)
- used **SMOTE(synthetic Minority Over-sampling Technique)** to balance the dataset

## Libraries Used:
- **Data Processing:** `pandas`,`numpy`,`sklearn.preprocessing`
- **Modelling:** `imblearn.pipeline`,`XGBoost`, `RandomForest`,`SVM`
- **Evaluation:** `classification_report`,`roc_auc_score`,`accuracy_score`

## Modelling
Different machine learning models were tested using a **Pipeline** with cross-validation and hyperparameter tuning

**Models Evaluated**
1. Logistic Regression(Baseline)
2. Support Vector Machine(SVM)
3. Decision Tree
4. Random Forest
5. XGBoost(best performing model)

## Tuning and Optimization:
- **GridSearchCV** was used to fine-tune `XGBoost` hyperparameters(`max_depth,n_estimators,learning_rate,subsample`)

## Evaluation
**Model performance(Test Set)**
![image-3.png](attachment:image-3.png)

**Insights**
- **XGBoost was the best model** with an accuracy of 95.5% and an AUC score of 90.86 % meaning able to distinguish well between the churned and non churned customers
- **Random Forest** perfomed well but with slight overfitting
- ** Logistic Regression** was the least peforming 

## Validation Approach:
- Used **Train-Test Split**(80-20)

### 8.0 Conclusion and recommendations
### 8.1 Conclusion
### 1. key Factors Influencing churn:
- **Total Day minutes**(Higher Usage linked to churn)
- **Customer service Calls** (Frequent calls indicate dissatisfaction)
- **International Plan** (customers with international plans has a higher churn rate)
### 2. Model performance:
- **XGBoost Outperformed all other models**,achieving the highest accuracy and AUC and hence we choose it as the best model to predict customer churn
- Feature engineering(log transformation and scaling) help reduce bias
### 3. Business Takeaways:
- Customers making longer calls during the day are more likely to churn
-Frequent calls to customer service indicate dissatisfaction and potential churn
- The international Plan segment is at higher risk, requiring retention strategies

### 8.2 Recommendations
### 1. Improve Customer Retention Strategies
- Offer **loyalty incentives** (discounts,special plans) for high-usage customers
- **Improve customer service** response time to reduce frustration
- **Targeted engagement** for customers with **international plans** (r.g better pricing, exclusive offers)

### 2. Use predictive Model in business operations
- Deploy the **XGBoost model** in production to `predict churn in real time`
- Develop an **automated alert system** to flag high-risk customers.
- Implement a **Personalized retention strategy** for flagged customers

### 3. Further improvements
- collect **additional data** (e.g, customer satisfaction surveys)
- Experiment with **deep learning models** for even better performance
- **Monitor model performance** and retrain regularly
