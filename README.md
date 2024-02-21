# Customer-Retention-Supervised-Learning
### This repo contains files related to a Final Project completed for BANA 4080 (Data Mining) at the University of Cincinnati

In this project various supervised machine learning binary classification models were used to predict the 
retention of customers for a Telecommunication company. The data file (customer_retention.csv) used to train 
these models consists of 7000 observations containing 19 features (predictor variables):

* Gender: Whether the customer is a male or a female
* SeniorCitizen: Whether the customer is a senior citizen or not (1, 0)
* Partner: Whether the customer has a partner or not (Yes, No)
* Dependents: Whether the customer has dependents or not (Yes, No)
* Tenure: Number of months the customer has stayed with the company
* PhoneService: Whether the customer has a phone service or not (Yes, No)
* MultipleLines: Whether the customer has multiple lines or not (Yes, No, No phone service)
* InternetService: Customer’s internet service provider (DSL, Fiber optic, No)
* OnlineSecurity: Whether the customer has online security or not (Yes, No, No internet service)
* OnlineBackup: Whether the customer has online backup or not (Yes, No, No internet service)
* DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service)
* TechSupport: Whether the customer has tech support or not (Yes, No, No internet service)
* StreamingTV : Whether the customer has streaming TV or not (Yes, No, No internet service)
* StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No internet service)
* Contract: The contract term of the customer (Month-to-month, One year, Two year)
* PaperlessBilling: Whether the customer has paperless billing or not (Yes, No)
* PaymentMethod: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
* MonthlyCharges: The amount charged to the customer monthly
* TotalCharges: The total amount charged to the customer
The target (response variable) in the data set is Status, which indicates if a customer is current or left their plan.

Logistic Regression, Multivariate Adaptive Regression Splines (MARS), Decision Tree, Bagged Tree, and 
Random Forest models were employed to fit the data, find the most important variables relating to customer
retention, and identify customers who are likely to leave. 
