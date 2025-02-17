# BigMart Sales Prediction

## Project Overview
BigMart Sales Prediction aims to forecast sales based on various product and store attributes. This project uses machine learning techniques to build predictive models and improve sales forecasting accuracy.

## Dataset
The dataset contains sales data for different items across multiple outlets, with features such as:
- `Item_Identifier`: Unique ID for products
- `Item_Weight`: Weight of the product
- `Item_Fat_Content`: Fat level of the product
- `Item_Visibility`: Visibility percentage of the product
- `Item_Type`: Category of the product
- `Outlet_Identifier`: Unique ID for store outlets
- `Outlet_Establishment_Year`: Year when the outlet was established
- `Outlet_Size`: Size of the outlet (Small/Medium/Large)
- `Outlet_Location_Type`: Tier of the outlet location
- `Outlet_Type`: Type of the outlet (Grocery/ Supermarket)
- `Item_Outlet_Sales`: Target variable representing sales

## Preprocessing and Feature Engineering
1. **Handling Missing Values:**
   - Filling `Item_Weight` using the median weight per `Item_Identifier`
   - Filling `Outlet_Size` based on `Outlet_Type`
2. **Feature Transformation:**
   - Replacing zero `Item_Visibility` with the mean visibility per `Item_Identifier`
   - One-hot encoding categorical variables
   - Feature scaling using `StandardScaler` or `MinMaxScaler`
3. **Train-Test Split:**
   - Training: 80%
   - Testing: 20%

## Model Implementation
The following machine learning models are tested:
1. **Baseline Model**: Mean sales per (`Item_Identifier`, `Outlet_Identifier`)
2. **Linear Regression**
3. **Ridge Regression**
4. **Lasso Regression**
5. **Decision Tree Regressor**
6. **Random Forest Regressor**
7. **Gradient Boosting Regressor**
8. **LightGBM Regressor**
9. **XGBoost Regressor**

## Model Evaluation Metrics
The following metrics are used to assess model performance:
- **Root Mean Squared Error (RMSE)**: Measures prediction error magnitude
- **R² Score**: Measures variance explained by the model
- **Mean Absolute Error (MAE)**: Measures absolute prediction error
- **Cross-Validation Scores**: Used for robustness check

## Hyperparameter Tuning
- **RandomizedSearchCV & GridSearchCV** are used to find optimal hyperparameters for boosting models (LGBM, XGBoost, Gradient Boosting).

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn
   ```
2. Run the Python script:
   ```bash
   python BigMartSalesPrediction.py
   ```
3. Predictions will be stored in the `results` folder.

## Future Enhancements
- Feature selection using SHAP or Recursive Feature Elimination
- Stacking ensemble models for improved accuracy
- Deploying the model as an API

---
**Author**: [Sharjeel Ahmed]

