# datascience_salary
Data Science jobs salary prediction based on Glassdoor data

# Data Cleaning
  #salary:
    1. Min & Max separate columns
    2. Remove glassdoor est
#Company Name
    1. Keep name only, remove rating
#Location
    1. City and State separate
#Founded
    1. Company Age
#Job Description
    1. Parse JD for keywords

# Salary Parse
1. Removed '-1' values in Salary Estimate
2. Removed 'glassdoor estimates'
3. Removed k and '$' sign in salaries
4. Added a column for 'per hour' and 'employer provided salary' with 1 if true or 0 if false
5. Removed 'per hour' and 'employer provided salary'
6. Split min and max salary into two columns

# Company Name 
  ### Has company name and rating together (could have done straight split by using delimiter as '\n'
1. Replaced '\n' with ","
2. Split company name and rating taking only company name and put it into column company txt

# Location
1. Separate State and city using "," as delimiter in state and city columns using 'split' <br/>
Note: One record had two "," "xxxx, los Angeles, CA" - Manually replaced Los Angeles with CA using 'replace'
2. Added a column same state as hq

# Founded
There were a few values with '-1', so replaced them with NAN and for rest of records subtracted from current year to get the age in a separate column company_age.

# Job Description Parsing
Keywords to search for: (convert everything to lower case before searching)
  1. Python
  2. SQL
  3. Excel
  4. Spark
  5. Apache
  6. R or R studio or R-Studio
  7. tensorflow or tensor or tesner flow
  8. Jupyter
  9. git
  10. github
  11. pandas
  12. numpy
  13. aws
  14. scikit or scikit-learn or scikit learn
  15. azure
  16. cloud or google cloud
  17. linux

# Model Building
First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.

I tried three different models:

Multiple Linear Regression – Baseline for the model
Lasso Regression – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
Random Forest – Again, with the sparsity associated with the data, I thought that this would be a good fit.
Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets.

Random Forest : MAE = 11.22
Linear Regression: MAE = 18.86
Ridge Regression: MAE = 19.67


    
