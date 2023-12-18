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

# Job Title
# Seniority
# JD Length
# Competitor Count
# Per hour wage to annual wage
# Remove new line from job title


    
