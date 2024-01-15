# DataScience Salary Prediction

## Overview

This project focuses on predicting Data Science job salaries based on Glassdoor data. The process involves data cleaning, salary parsing, and model building using various machine-learning techniques.

**MLOps Initiative:**
In a recent project, I led a comprehensive MLOps initiative to create a salary estimator using Python. This involved scraping data from Glassdoor, amassing approximately 4,000 job entries, and employing Selenium in Python to curate a dataset with 28 critical columns.

**Data Cleaning and Feature Engineering:**
The project demanded meticulous data cleaning, ensuring data accuracy and coherence by addressing missing values and inconsistencies. I performed feature engineering to enhance the dataset's predictive power, optimizing it for subsequent machine learning model development.

**Model Selection and Optimization:**
To construct the salary estimator, I employed Linear Regression, Random Forest, and Lasso Regression. I developed a tool that estimates salaries for jobs, scraped over 4000+ job descriptions, and engineered features to quantify the value companies place on tools like Python, AWS, Spark, and R. The model achieved an accuracy of 92.34%. Furthermore, I utilized XG Boost to optimize the model's cross-validation, ensuring robust performance and accuracy.

## Data Cleaning

### Salary
- Separated minimum and maximum salary into individual columns.
- Removed 'glassdoor estimates'.
- Processed salary values by removing 'k' and '$' signs.
- Added columns for 'per hour' and 'employer-provided salary' with binary indicators.
- Split min and max salary into two separate columns.

### Company Name
- Extracted company names by removing rating and keeping only the name.
- Split company name and rating, storing only the company name in a dedicated column.

### Location
- Separated city and state using ',' as a delimiter.
- Addressed an exception where a record had two ',' by manually replacing the inconsistency.
- Added a column indicating whether the job location is in the same state as the company headquarters.

### Founded
- Converted '-1' values to NaN.
- Calculated company age by subtracting the founding year from the current year.

### Job Description Parsing
- Extracted keywords from job descriptions, converting everything to lowercase.
- Keywords include Python, SQL, Excel, Spark, Apache, R, TensorFlow, Jupyter, Git, GitHub, Pandas, NumPy, AWS, Scikit-learn, Azure, Cloud, Google Cloud, Linux.

## Model Building

### Data Transformation
- Converted categorical variables into dummy variables.
- Split the data into training and testing sets with a 20% test size.

### Models Evaluated
1. Multiple Linear Regression – Baseline model.
2. Lasso Regression – Effective for sparse data due to numerous categorical variables.
3. Random Forest – Suitable for the sparsity associated with the data.
4. XG Boost – Optimized the model's cross-validation for enhanced performance.

### Model Performance
- Evaluated models using Mean Absolute Error (MAE).
- Random Forest outperformed other models:
  - Random Forest: MAE = 11.22
  - Linear Regression: MAE = 18.86
  - Ridge Regression: MAE = 19.67
