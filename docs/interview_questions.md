# Interview & Viva Questions: Health Insurance Premium Prediction

A curated set of 22 practical Machine Learning interview and academic viva questions tailored to this project.

---

### 1. What is regression in Machine Learning, and why is this project a regression problem?
**Answer:** Regression is a supervised learning task where the model predicts a continuous numerical value rather than a discrete category. Because our target variable (`charges`) is a continuous dollar amount representing annual medical costs, this is a regression problem.

### 2. How does Linear Regression work, and what are its main limitations on this dataset?
**Answer:** Linear Regression predicts the target as a weighted sum of input features plus an intercept, choosing weights that minimize the sum of squared residuals (Ordinary Least Squares). Its main limitation here is that health insurance charges exhibit strong non-linear interactions—especially between smoking status and BMI—which a plain linear model misses unless explicit interaction features are engineered.

### 3. How does a Decision Tree Regressor make predictions?
**Answer:** A Decision Tree recursively splits the dataset into smaller subsets based on feature thresholds that minimize target variance (MSE). At prediction time, a new observation follows the decision rules down to a leaf node, and the predicted value is the mean of the training targets in that leaf.

### 4. What is a Random Forest Regressor, and why does it usually outperform a single Decision Tree?
**Answer:** A Random Forest trains many independent Decision Trees on random bootstrap samples of the training data and random subsets of features at each split, then averages their predictions (bagging). Averaging across decorrelated trees dramatically reduces variance and overfitting compared to a single deep tree.

### 5. How does Gradient Boosting Regressor differ from Random Forest?
**Answer:** Random Forest builds deep trees in parallel and averages them to reduce variance. Gradient Boosting builds shallow trees sequentially, where each new tree is trained to predict the residual errors left by the ensemble so far, primarily reducing bias while controlling variance through a learning rate and tree depth.

### 6. What is feature engineering, and what engineered features did you explore in this project?
**Answer:** Feature engineering is the process of creating informative input variables from raw data using domain knowledge. In this project, we explored `bmi_category` (clinical BMI brackets), `age_group` (life-stage brackets), `smoker_bmi_interaction` (`bmi * is_smoker`), and `is_obese_smoker` (`smoker == 'yes'` and `bmi >= 30`), and compared model CV performance with and without them.

### 7. Why do we need categorical encoding, and why did we use One-Hot Encoding?
**Answer:** Scikit-learn regression algorithms require numerical inputs. Features like `sex`, `smoker`, and `region` are nominal categories with no natural numeric order. One-Hot Encoding creates binary indicator columns for each category so the model does not falsely assume an ordinal ranking (e.g., `southeast > northeast`). Using `drop="first"` avoids perfect multicollinearity in Linear Regression, and `handle_unknown="ignore"` prevents crashes on unseen categories.

### 8. Why do we scale numerical features, and which models benefit most from scaling?
**Answer:** Feature scaling (such as `StandardScaler`, which centers features to mean 0 and standard deviation 1) puts numerical features like `age`, `bmi`, and `children` on comparable scales. Distance-based and linear models (especially regularized linear models or gradient-based optimizers) benefit most; tree-based models (Decision Trees, Random Forest, Gradient Boosting) are invariant to monotonic feature scaling, though keeping scaling inside a unified `ColumnTransformer` keeps pipelines consistent.

### 9. What is data leakage, and how did you prevent it in this project?
**Answer:** Data leakage happens when information from the test set influences training or preprocessing (for example, fitting a `StandardScaler` or `SimpleImputer` on the full dataset before splitting). We prevented leakage by splitting the data 80/20 first and wrapping all preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) inside a scikit-learn `Pipeline` that is fit strictly on training folds.

### 10. What is overfitting, and how can you detect it?
**Answer:** Overfitting occurs when a model memorizes noise in the training set rather than learning generalizable patterns. You detect it when training error is very low (e.g., an unpruned Decision Tree with near-1.0 training R²) while cross-validation or test error is much higher.

### 11. What is underfitting, and how do you address it?
**Answer:** Underfitting occurs when a model is too simple to capture the underlying patterns in the data, resulting in high error on both training and test sets. You address it by using more expressive models (e.g., moving from plain Linear Regression to Gradient Boosting), adding informative interaction features, or reducing overly strict regularization.

### 12. Explain MAE, MSE, RMSE, and R² Score.
**Answer:**
- **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual values (in dollars); easy to interpret and less sensitive to outliers.
- **MSE (Mean Squared Error):** Average squared difference; penalizes large errors heavily.
- **RMSE (Root Mean Squared Error):** Square root of MSE, bringing the error back into dollars while still penalizing large misses more than MAE.
- **R² Score:** Fraction of variance in the target variable explained by the model (`1 - SS_res / SS_tot`).

### 13. Why do we evaluate using both MAE and RMSE instead of only one metric?
**Answer:** MAE tells you the typical error magnitude in dollars, while RMSE highlights whether the model is making occasional very large errors. Comparing both reveals if a handful of high-charge cases are skewing model accuracy.

### 14. What is cross-validation, and why is it better than a single train-validation split?
**Answer:** In k-fold cross-validation (5-fold in this project), the training set is split into `k` equal folds; the model is trained on `k-1` folds and validated on the remaining fold `k` times, and scores are averaged. This gives a more reliable estimate of generalization performance and reduces sensitivity to a lucky or unlucky single split.

### 15. What is hyperparameter tuning, and how does GridSearchCV work?
**Answer:** Hyperparameters are configuration settings chosen before training (like `max_depth`, `n_estimators`, and `learning_rate`). `GridSearchCV` exhaustively evaluates every combination in a defined parameter grid using k-fold cross-validation on the training set only, then refits the best-scoring combination on the full training set.

### 16. Why must the test set remain completely untouched during model selection and hyperparameter tuning?
**Answer:** If you tune hyperparameters or select models using test-set metrics, the test set effectively becomes part of the training loop, producing an optimistically biased performance estimate that will not hold on truly unseen data.

### 17. What is a scikit-learn Pipeline, and why is it important for deployment?
**Answer:** A `Pipeline` chains feature engineering, preprocessing (`ColumnTransformer`), and the estimator into a single object. During deployment, you pass raw user inputs directly to `pipeline.predict()`, guaranteeing that exact same transformations and encodings used during training are applied at inference time with zero training-serving skew.

### 18. Why did we use Joblib to save the trained model?
**Answer:** `joblib` is optimized for serializing Python objects containing large NumPy arrays (such as scikit-learn models and pipelines) more efficiently than standard `pickle`. Saving the entire `Pipeline` lets the Streamlit app load and serve predictions instantly without retraining.

### 19. What is feature importance, and how do you interpret it?
**Answer:** For tree-based models (`RandomForestRegressor`, `GradientBoostingRegressor`), feature importance measures how much each feature reduces impurity (variance) across all splits in the ensemble. It helps identify the strongest cost drivers—such as smoking status, BMI interaction, and age—though correlated features can split importance among themselves.

### 20. What do residual plots tell you about a regression model?
**Answer:** A residual plot graphs `Actual - Predicted` errors against predicted values. A well-behaved model shows residuals scattered randomly around zero with roughly constant spread. Patterns, curves, or funnel shapes indicate missed non-linearities, heteroscedasticity, or subgroups where the model systematically under- or over-predicts.

### 21. Why did we keep high-charge outliers in the dataset instead of deleting them?
**Answer:** In medical insurance data, very high charges (e.g., $40,000–$60,000+) usually belong to real high-risk individuals such as older obese smokers rather than data entry errors. Removing them would bias the model to underestimate costs for real high-risk users.

### 22. How does the Streamlit application avoid slowing down on every user interaction?
**Answer:** The Streamlit app loads pre-trained artifacts (`models/best_model.joblib` and saved CSV/JSON/PNG outputs) instead of retraining models on startup, and uses `@st.cache_resource` and `@st.cache_data` so the model and evaluation files are loaded from disk only once per session.
