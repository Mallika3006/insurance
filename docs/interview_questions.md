# Interview & Viva Questions — Indian Health Insurance Premium Prediction

### 1. Why is this a regression problem?
The target PremiumPrice is a continuous numerical amount, so the task is regression rather than classification.

### 2. Which dataset is used?
The public Medical Insurance Premium Prediction dataset from Kaggle, with 986 records, ten input attributes and PremiumPrice as the target.

### 3. How is this different from the original project?
The original project used a US dataset with smoking, sex, children and US region. This version uses age, health indicators, height, weight and surgery history.

### 4. How is BMI calculated?
BMI = weight in kilograms / (height in metres × height in metres). Height is converted from centimetres to metres first.

### 5. Why engineer BMI?
It combines height and weight into a commonly used body-measurement indicator that may help the model identify patterns.

### 6. Which algorithms are compared?
Linear Regression, Decision Tree Regressor, Random Forest Regressor and Gradient Boosting Regressor.

### 7. Why use a Pipeline?
A Pipeline applies the same feature engineering and preprocessing during training and prediction, reducing training-serving inconsistencies and data leakage.

### 8. What is cross-validation?
Five-fold cross-validation trains and validates the model across five partitions of the training set, giving a more stable model comparison.

### 9. What do MAE, MSE, RMSE and R² measure?
MAE is average absolute error, MSE is average squared error, RMSE is the square root of MSE and R² measures explained target variance. Monetary error metrics use the target's units (₹).

### 10. Is the output an actual Indian insurance quote?
No. It is a model estimate from a small public dataset. Real insurer prices depend on policy, sum insured, location, age rules, exclusions and underwriting.
