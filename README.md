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
Removed '-1' values in Salary Estimate
Removed 'glassdoor estimates'
Removed k and '$' sign in salaries
Added a column for 'per hour' and 'employer provided salary' with 1 if true or 0 if false
Removed 'per hour' and 'employer provided salary'
Split min and max salary into two columns

# Company Name -  Has company name and rating together (could have done straight split by using delimiter as '\n'
Replaced '\n' with , 
Split company name and rating taking only company name and put it into column company txt

# Location

    
