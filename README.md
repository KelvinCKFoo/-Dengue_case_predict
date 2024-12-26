

---

# Dengue Fever Case Prediction

This repository contains a machine learning pipeline designed to predict Dengue fever cases using historical weather and environmental data. The project leverages various data preprocessing, exploratory data analysis (EDA), and machine learning techniques to model and predict Dengue case counts.

## Features
- **Exploratory Data Analysis (EDA)**: 
  - Visualizations including correlation heatmaps, box plots, and line charts.
  - Statistical summaries of the dataset.
- **Data Preprocessing**: 
  - Handling missing values using mean/median imputation.
  - Feature engineering (lagged features, rolling averages, and encoding).
  - Data normalization and feature selection.
- **Machine Learning Models**:
  - **Decision Tree Regressor**: For baseline regression.
  - **Support Vector Regressor (SVR)**: With hyperparameter tuning for optimized predictions.
  - **Random Forest Regressor**: For robust modeling of nonlinear dependencies.
  - **Dual-Linear Regression Model**: Incorporates seasonal trends and smoothing for enhanced performance.
- **Hyperparameter Optimization**:
  - Grid search for parameter tuning in models like SVR and Random Forest.
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-Squared.

## Workflow
1. **Data Loading**:
   - Training and testing datasets are loaded and merged for analysis.
2. **EDA**:
   - Generate visualizations to understand data distributions and correlations.
3. **Preprocessing**:
   - Handle missing values, normalize features, and engineer lagged and rolling average features.
   - Encode categorical variables and prepare data for machine learning models.
4. **Modeling**:
   - Train models like Decision Tree, Random Forest, and SVR.
   - Use feature selection and weight adjustments to optimize predictions.
5. **Prediction and Evaluation**:
   - Evaluate model performance using test datasets.
   - Visualize actual vs. predicted cases to assess accuracy.
6. **Submission**:
   - Generate CSV files with predictions for submission to the evaluation platform.

## Results
- **Dual-Linear Regression**:
  - Achieved the best web evaluation score of **18.3269**.
- **Random Forest Regressor**:
  - Effectively modeled complex interactions with competitive local MAE scores.

## Repository Structure
- `Main.ipynb`: Core notebook containing the pipeline for data preprocessing, modeling, and evaluation.
- `dengue_features_train.csv`, `dengue_labels_train.csv`: Training datasets.
- `dengue_features_test.csv`: Test dataset.
- `Submission.csv`: Template for prediction submission.
- `Submission_RandomForest.csv`: Final submission file for Random Forest predictions.
- `Submission_dual_liner.csv`: Final submission file for Dual-Linear Regression predictions.

## Requirements
- Python 3.8+
- Libraries: 
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `sklearn` (Scikit-learn)

Install required dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/KelvinCKFoo/-Dengue_case_predict.git
   
   ```
2. Navigate to the project directory:
   ```bash
   cd dengue-prediction
   ```
3. Run the Jupyter Notebook or execute the provided Python script:
   ```bash
   jupyter notebook Main.ipynb
   ```
