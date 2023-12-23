```python
import pandas as pd
```


```python
df = pd.read_csv('glassdoor_jobs.csv')
```


```python
#pd.set_option('display.max_rows', None)
pd.reset_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

```

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


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 956 entries, 0 to 955
    Data columns (total 15 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Unnamed: 0         956 non-null    int64  
     1   Job Title          956 non-null    object 
     2   Salary Estimate    956 non-null    object 
     3   Job Description    956 non-null    object 
     4   Rating             956 non-null    float64
     5   Company Name       956 non-null    object 
     6   Location           956 non-null    object 
     7   Headquarters       956 non-null    object 
     8   Size               956 non-null    object 
     9   Founded            956 non-null    int64  
     10  Type of ownership  956 non-null    object 
     11  Industry           956 non-null    object 
     12  Sector             956 non-null    object 
     13  Revenue            956 non-null    object 
     14  Competitors        956 non-null    object 
    dtypes: float64(1), int64(2), object(12)
    memory usage: 112.2+ KB
    


```python
df.isna().sum()
```




    Unnamed: 0           0
    Job Title            0
    Salary Estimate      0
    Job Description      0
    Rating               0
    Company Name         0
    Location             0
    Headquarters         0
    Size                 0
    Founded              0
    Type of ownership    0
    Industry             0
    Sector               0
    Revenue              0
    Competitors          0
    dtype: int64




```python
(df['Salary Estimate'] == '-1').sum()
```




    214




```python
df = df[df['Salary Estimate']!='-1']
```

# Regex

### Cleaning Salary


```python
regex_df = df.copy()
```


```python
#Glassdoor estimate removal
salary = regex_df['Salary Estimate'].apply(lambda x:x.split('(')[0])
```


```python
regex_df['per_hour'] = salary.apply(lambda x: 1 if "per hour" in x.lower() else 0)
```


```python
regex_df['employer_provided'] = salary.apply(lambda x: 1 if "employer provided salary:" in x.lower() else 0)
```


```python
#'k' and '$' removal
remove_kd = salary.apply(lambda x:x.replace('K','').replace('$', ''))
```


```python
remove_text = remove_kd.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:', ''))
```


```python
regex_df['min salary'] = remove_text.apply(lambda x: x.split('-')[0])
```


```python
regex_df['max_salary'] =  remove_text.apply(lambda x:x.split('-')[1])
```

### Company Name Parse


```python
remove_n = regex_df['Company Name'].apply(lambda x: x.replace('\n', ','))
```


```python
regex_df['company_txt'] = remove_n.apply(lambda x: x.split(',')[0])
```

# Location


```python
extract_state = regex_df['Location'].apply(lambda x: x.split(',')[1])
extract_state = extract_state.apply(lambda x: x.replace('Los Angeles', 'CA'))
```


```python
extract_city = regex_df['Location'].apply(lambda x: x.split(',')[0])
```


```python
regex_df['State'] = extract_state
regex_df['City'] = extract_city
```


```python
# It was throwing an error with x.split(',')[1], so used [-1] to get the states values.
extract_hq_state = regex_df['Headquarters'].apply(lambda x: x.split(',')[-1])
```


```python
#Test since above code wasn't working, it worked with [-1]
#for x in regex_df['Headquarters']:
    #split_result = x.split(',')[1]
    #print((split_result))
```


```python
temp_df = pd.DataFrame({'extract_state' : extract_state, 'extract_hq_state' : extract_hq_state})

```


```python
same_state_hq = temp_df.apply(lambda x: 1 if x.extract_state == x.extract_hq_state else 0, axis =1)

```


```python
regex_df['same_state_hq'] = same_state_hq
```

# Founded


```python
from datetime import datetime
```


```python
year = datetime.now().year
```


```python
company_age = []
for x in regex_df.Founded:
    if x == -1:
        company_age.append('NAN')
    else:
        age = year - x
        company_age.append(age)

```


```python
regex_df['company_age'] = company_age
```


```python
regex_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Job Title</th>
      <th>Salary Estimate</th>
      <th>Job Description</th>
      <th>Rating</th>
      <th>Company Name</th>
      <th>Location</th>
      <th>Headquarters</th>
      <th>Size</th>
      <th>Founded</th>
      <th>Type of ownership</th>
      <th>Industry</th>
      <th>Sector</th>
      <th>Revenue</th>
      <th>Competitors</th>
      <th>per_hour</th>
      <th>employer_provided</th>
      <th>min salary</th>
      <th>max_salary</th>
      <th>company_txt</th>
      <th>State</th>
      <th>City</th>
      <th>same_state_hq</th>
      <th>company_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Data Scientist</td>
      <td>$53K-$91K (Glassdoor est.)</td>
      <td>Data Scientist\nLocation: Albuquerque, NM\nEdu...</td>
      <td>3.8</td>
      <td>Tecolote Research\n3.8</td>
      <td>Albuquerque, NM</td>
      <td>Goleta, CA</td>
      <td>501 to 1000 employees</td>
      <td>1973</td>
      <td>Company - Private</td>
      <td>Aerospace &amp; Defense</td>
      <td>Aerospace &amp; Defense</td>
      <td>$50 to $100 million (USD)</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>53</td>
      <td>91</td>
      <td>Tecolote Research</td>
      <td>NM</td>
      <td>Albuquerque</td>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Healthcare Data Scientist</td>
      <td>$63K-$112K (Glassdoor est.)</td>
      <td>What You Will Do:\n\nI. General Summary\n\nThe...</td>
      <td>3.4</td>
      <td>University of Maryland Medical System\n3.4</td>
      <td>Linthicum, MD</td>
      <td>Baltimore, MD</td>
      <td>10000+ employees</td>
      <td>1984</td>
      <td>Other Organization</td>
      <td>Health Care Services &amp; Hospitals</td>
      <td>Health Care</td>
      <td>$2 to $5 billion (USD)</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>63</td>
      <td>112</td>
      <td>University of Maryland Medical System</td>
      <td>MD</td>
      <td>Linthicum</td>
      <td>1</td>
      <td>39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Data Scientist</td>
      <td>$80K-$90K (Glassdoor est.)</td>
      <td>KnowBe4, Inc. is a high growth information sec...</td>
      <td>4.8</td>
      <td>KnowBe4\n4.8</td>
      <td>Clearwater, FL</td>
      <td>Clearwater, FL</td>
      <td>501 to 1000 employees</td>
      <td>2010</td>
      <td>Company - Private</td>
      <td>Security Services</td>
      <td>Business Services</td>
      <td>$100 to $500 million (USD)</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>90</td>
      <td>KnowBe4</td>
      <td>FL</td>
      <td>Clearwater</td>
      <td>1</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Data Scientist</td>
      <td>$56K-$97K (Glassdoor est.)</td>
      <td>*Organization and Job ID**\nJob ID: 310709\n\n...</td>
      <td>3.8</td>
      <td>PNNL\n3.8</td>
      <td>Richland, WA</td>
      <td>Richland, WA</td>
      <td>1001 to 5000 employees</td>
      <td>1965</td>
      <td>Government</td>
      <td>Energy</td>
      <td>Oil, Gas, Energy &amp; Utilities</td>
      <td>$500 million to $1 billion (USD)</td>
      <td>Oak Ridge National Laboratory, National Renewa...</td>
      <td>0</td>
      <td>0</td>
      <td>56</td>
      <td>97</td>
      <td>PNNL</td>
      <td>WA</td>
      <td>Richland</td>
      <td>1</td>
      <td>58</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Data Scientist</td>
      <td>$86K-$143K (Glassdoor est.)</td>
      <td>Data Scientist\nAffinity Solutions / Marketing...</td>
      <td>2.9</td>
      <td>Affinity Solutions\n2.9</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>51 to 200 employees</td>
      <td>1998</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>Unknown / Non-Applicable</td>
      <td>Commerce Signals, Cardlytics, Yodlee</td>
      <td>0</td>
      <td>0</td>
      <td>86</td>
      <td>143</td>
      <td>Affinity Solutions</td>
      <td>NY</td>
      <td>New York</td>
      <td>1</td>
      <td>25</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>950</th>
      <td>950</td>
      <td>Sr Scientist, Immuno-Oncology - Oncology</td>
      <td>$58K-$111K (Glassdoor est.)</td>
      <td>Site Name: USA - Massachusetts - Cambridge\nPo...</td>
      <td>3.9</td>
      <td>GSK\n3.9</td>
      <td>Cambridge, MA</td>
      <td>Brentford, United Kingdom</td>
      <td>10000+ employees</td>
      <td>1830</td>
      <td>Company - Public</td>
      <td>Biotech &amp; Pharmaceuticals</td>
      <td>Biotech &amp; Pharmaceuticals</td>
      <td>$10+ billion (USD)</td>
      <td>Pfizer, AstraZeneca, Merck</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
      <td>111</td>
      <td>GSK</td>
      <td>MA</td>
      <td>Cambridge</td>
      <td>0</td>
      <td>193</td>
    </tr>
    <tr>
      <th>951</th>
      <td>951</td>
      <td>Senior Data Engineer</td>
      <td>$72K-$133K (Glassdoor est.)</td>
      <td>THE CHALLENGE\nEventbrite has a world-class da...</td>
      <td>4.4</td>
      <td>Eventbrite\n4.4</td>
      <td>Nashville, TN</td>
      <td>San Francisco, CA</td>
      <td>1001 to 5000 employees</td>
      <td>2006</td>
      <td>Company - Public</td>
      <td>Internet</td>
      <td>Information Technology</td>
      <td>$100 to $500 million (USD)</td>
      <td>See Tickets, TicketWeb, Vendini</td>
      <td>0</td>
      <td>0</td>
      <td>72</td>
      <td>133</td>
      <td>Eventbrite</td>
      <td>TN</td>
      <td>Nashville</td>
      <td>0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>952</th>
      <td>952</td>
      <td>Project Scientist - Auton Lab, Robotics Institute</td>
      <td>$56K-$91K (Glassdoor est.)</td>
      <td>The Auton Lab at Carnegie Mellon University is...</td>
      <td>2.6</td>
      <td>Software Engineering Institute\n2.6</td>
      <td>Pittsburgh, PA</td>
      <td>Pittsburgh, PA</td>
      <td>501 to 1000 employees</td>
      <td>1984</td>
      <td>College / University</td>
      <td>Colleges &amp; Universities</td>
      <td>Education</td>
      <td>Unknown / Non-Applicable</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>56</td>
      <td>91</td>
      <td>Software Engineering Institute</td>
      <td>PA</td>
      <td>Pittsburgh</td>
      <td>1</td>
      <td>39</td>
    </tr>
    <tr>
      <th>953</th>
      <td>953</td>
      <td>Data Science Manager</td>
      <td>$95K-$160K (Glassdoor est.)</td>
      <td>Data Science ManagerResponsibilities:\n\nOvers...</td>
      <td>3.2</td>
      <td>Numeric, LLC\n3.2</td>
      <td>Allentown, PA</td>
      <td>Chadds Ford, PA</td>
      <td>1 to 50 employees</td>
      <td>-1</td>
      <td>Company - Private</td>
      <td>Staffing &amp; Outsourcing</td>
      <td>Business Services</td>
      <td>$5 to $10 million (USD)</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>95</td>
      <td>160</td>
      <td>Numeric</td>
      <td>PA</td>
      <td>Allentown</td>
      <td>1</td>
      <td>NAN</td>
    </tr>
    <tr>
      <th>955</th>
      <td>955</td>
      <td>Research Scientist – Security and Privacy</td>
      <td>$61K-$126K (Glassdoor est.)</td>
      <td>Returning Candidate? Log back in to the Career...</td>
      <td>3.6</td>
      <td>Riverside Research Institute\n3.6</td>
      <td>Beavercreek, OH</td>
      <td>Arlington, VA</td>
      <td>501 to 1000 employees</td>
      <td>1967</td>
      <td>Nonprofit Organization</td>
      <td>Federal Agencies</td>
      <td>Government</td>
      <td>$50 to $100 million (USD)</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>61</td>
      <td>126</td>
      <td>Riverside Research Institute</td>
      <td>OH</td>
      <td>Beavercreek</td>
      <td>0</td>
      <td>56</td>
    </tr>
  </tbody>
</table>
<p>742 rows × 24 columns</p>
</div>



# JD Parse (Added each of these series to regex_df separately)


```python
regex_df['jd_python'] = regex_df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() or 'py' in x.lower() else 0)
```


```python
regex_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Job Title</th>
      <th>Salary Estimate</th>
      <th>Job Description</th>
      <th>Rating</th>
      <th>Company Name</th>
      <th>Location</th>
      <th>Headquarters</th>
      <th>Size</th>
      <th>Founded</th>
      <th>Type of ownership</th>
      <th>Industry</th>
      <th>Sector</th>
      <th>Revenue</th>
      <th>Competitors</th>
      <th>per_hour</th>
      <th>employer_provided</th>
      <th>min salary</th>
      <th>max_salary</th>
      <th>company_txt</th>
      <th>State</th>
      <th>City</th>
      <th>same_state_hq</th>
      <th>company_age</th>
      <th>jd_python</th>
      <th>jd_sql</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Data Scientist</td>
      <td>$53K-$91K (Glassdoor est.)</td>
      <td>Data Scientist\nLocation: Albuquerque, NM\nEdu...</td>
      <td>3.8</td>
      <td>Tecolote Research\n3.8</td>
      <td>Albuquerque, NM</td>
      <td>Goleta, CA</td>
      <td>501 to 1000 employees</td>
      <td>1973</td>
      <td>Company - Private</td>
      <td>Aerospace &amp; Defense</td>
      <td>Aerospace &amp; Defense</td>
      <td>$50 to $100 million (USD)</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>53</td>
      <td>91</td>
      <td>Tecolote Research</td>
      <td>NM</td>
      <td>Albuquerque</td>
      <td>0</td>
      <td>50</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Healthcare Data Scientist</td>
      <td>$63K-$112K (Glassdoor est.)</td>
      <td>What You Will Do:\n\nI. General Summary\n\nThe...</td>
      <td>3.4</td>
      <td>University of Maryland Medical System\n3.4</td>
      <td>Linthicum, MD</td>
      <td>Baltimore, MD</td>
      <td>10000+ employees</td>
      <td>1984</td>
      <td>Other Organization</td>
      <td>Health Care Services &amp; Hospitals</td>
      <td>Health Care</td>
      <td>$2 to $5 billion (USD)</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>63</td>
      <td>112</td>
      <td>University of Maryland Medical System</td>
      <td>MD</td>
      <td>Linthicum</td>
      <td>1</td>
      <td>39</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Data Scientist</td>
      <td>$80K-$90K (Glassdoor est.)</td>
      <td>KnowBe4, Inc. is a high growth information sec...</td>
      <td>4.8</td>
      <td>KnowBe4\n4.8</td>
      <td>Clearwater, FL</td>
      <td>Clearwater, FL</td>
      <td>501 to 1000 employees</td>
      <td>2010</td>
      <td>Company - Private</td>
      <td>Security Services</td>
      <td>Business Services</td>
      <td>$100 to $500 million (USD)</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>90</td>
      <td>KnowBe4</td>
      <td>FL</td>
      <td>Clearwater</td>
      <td>1</td>
      <td>13</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Data Scientist</td>
      <td>$56K-$97K (Glassdoor est.)</td>
      <td>*Organization and Job ID**\nJob ID: 310709\n\n...</td>
      <td>3.8</td>
      <td>PNNL\n3.8</td>
      <td>Richland, WA</td>
      <td>Richland, WA</td>
      <td>1001 to 5000 employees</td>
      <td>1965</td>
      <td>Government</td>
      <td>Energy</td>
      <td>Oil, Gas, Energy &amp; Utilities</td>
      <td>$500 million to $1 billion (USD)</td>
      <td>Oak Ridge National Laboratory, National Renewa...</td>
      <td>0</td>
      <td>0</td>
      <td>56</td>
      <td>97</td>
      <td>PNNL</td>
      <td>WA</td>
      <td>Richland</td>
      <td>1</td>
      <td>58</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Data Scientist</td>
      <td>$86K-$143K (Glassdoor est.)</td>
      <td>Data Scientist\nAffinity Solutions / Marketing...</td>
      <td>2.9</td>
      <td>Affinity Solutions\n2.9</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>51 to 200 employees</td>
      <td>1998</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>Unknown / Non-Applicable</td>
      <td>Commerce Signals, Cardlytics, Yodlee</td>
      <td>0</td>
      <td>0</td>
      <td>86</td>
      <td>143</td>
      <td>Affinity Solutions</td>
      <td>NY</td>
      <td>New York</td>
      <td>1</td>
      <td>25</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>950</th>
      <td>950</td>
      <td>Sr Scientist, Immuno-Oncology - Oncology</td>
      <td>$58K-$111K (Glassdoor est.)</td>
      <td>Site Name: USA - Massachusetts - Cambridge\nPo...</td>
      <td>3.9</td>
      <td>GSK\n3.9</td>
      <td>Cambridge, MA</td>
      <td>Brentford, United Kingdom</td>
      <td>10000+ employees</td>
      <td>1830</td>
      <td>Company - Public</td>
      <td>Biotech &amp; Pharmaceuticals</td>
      <td>Biotech &amp; Pharmaceuticals</td>
      <td>$10+ billion (USD)</td>
      <td>Pfizer, AstraZeneca, Merck</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
      <td>111</td>
      <td>GSK</td>
      <td>MA</td>
      <td>Cambridge</td>
      <td>0</td>
      <td>193</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>951</th>
      <td>951</td>
      <td>Senior Data Engineer</td>
      <td>$72K-$133K (Glassdoor est.)</td>
      <td>THE CHALLENGE\nEventbrite has a world-class da...</td>
      <td>4.4</td>
      <td>Eventbrite\n4.4</td>
      <td>Nashville, TN</td>
      <td>San Francisco, CA</td>
      <td>1001 to 5000 employees</td>
      <td>2006</td>
      <td>Company - Public</td>
      <td>Internet</td>
      <td>Information Technology</td>
      <td>$100 to $500 million (USD)</td>
      <td>See Tickets, TicketWeb, Vendini</td>
      <td>0</td>
      <td>0</td>
      <td>72</td>
      <td>133</td>
      <td>Eventbrite</td>
      <td>TN</td>
      <td>Nashville</td>
      <td>0</td>
      <td>17</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>952</th>
      <td>952</td>
      <td>Project Scientist - Auton Lab, Robotics Institute</td>
      <td>$56K-$91K (Glassdoor est.)</td>
      <td>The Auton Lab at Carnegie Mellon University is...</td>
      <td>2.6</td>
      <td>Software Engineering Institute\n2.6</td>
      <td>Pittsburgh, PA</td>
      <td>Pittsburgh, PA</td>
      <td>501 to 1000 employees</td>
      <td>1984</td>
      <td>College / University</td>
      <td>Colleges &amp; Universities</td>
      <td>Education</td>
      <td>Unknown / Non-Applicable</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>56</td>
      <td>91</td>
      <td>Software Engineering Institute</td>
      <td>PA</td>
      <td>Pittsburgh</td>
      <td>1</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>953</th>
      <td>953</td>
      <td>Data Science Manager</td>
      <td>$95K-$160K (Glassdoor est.)</td>
      <td>Data Science ManagerResponsibilities:\n\nOvers...</td>
      <td>3.2</td>
      <td>Numeric, LLC\n3.2</td>
      <td>Allentown, PA</td>
      <td>Chadds Ford, PA</td>
      <td>1 to 50 employees</td>
      <td>-1</td>
      <td>Company - Private</td>
      <td>Staffing &amp; Outsourcing</td>
      <td>Business Services</td>
      <td>$5 to $10 million (USD)</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>95</td>
      <td>160</td>
      <td>Numeric</td>
      <td>PA</td>
      <td>Allentown</td>
      <td>1</td>
      <td>NAN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>955</th>
      <td>955</td>
      <td>Research Scientist – Security and Privacy</td>
      <td>$61K-$126K (Glassdoor est.)</td>
      <td>Returning Candidate? Log back in to the Career...</td>
      <td>3.6</td>
      <td>Riverside Research Institute\n3.6</td>
      <td>Beavercreek, OH</td>
      <td>Arlington, VA</td>
      <td>501 to 1000 employees</td>
      <td>1967</td>
      <td>Nonprofit Organization</td>
      <td>Federal Agencies</td>
      <td>Government</td>
      <td>$50 to $100 million (USD)</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>61</td>
      <td>126</td>
      <td>Riverside Research Institute</td>
      <td>OH</td>
      <td>Beavercreek</td>
      <td>0</td>
      <td>56</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>742 rows × 26 columns</p>
</div>




```python
regex_df.jd_python.value_counts()
```




    1    418
    0    324
    Name: jd_python, dtype: int64




```python
regex_df['jd_sql'] = regex_df['Job Description'].apply(lambda x: 1 if 'sql' in x.lower() or 'mysql' in x.lower() else 0)
```


```python
regex_df.jd_sql.value_counts()
```




    1    380
    0    362
    Name: jd_sql, dtype: int64




```python
regex_df['jd_excel'] = regex_df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() or 'msexcel' in x.lower() else 0)
```


```python
regex_df.jd_excel.value_counts()
```




    1    388
    0    354
    Name: jd_excel, dtype: int64




```python
regex_df['jd_spark'] = regex_df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
```


```python
regex_df.jd_spark.value_counts()
```




    0    575
    1    167
    Name: jd_spark, dtype: int64




```python
regex_df['jd_apache'] = regex_df['Job Description'].apply(lambda x: 1 if 'apache' in x.lower() else 0)
```


```python
regex_df.jd_apache.value_counts()
```




    0    709
    1     33
    Name: jd_apache, dtype: int64




```python
regex_df['jd_r'] = regex_df['Job Description'].apply(lambda x: 1 if 'r' in x.lower() or ' r studio' in x.lower() or 'r-studio' in x.lower()  else 0)
```


```python
regex_df.jd_r.value_counts()
```




    1    742
    Name: jd_r, dtype: int64




```python
regex_df['jd_tensor'] = regex_df['Job Description'].apply(lambda x: 1 if 'tensorflow' in x.lower() or 'tensor' in x.lower() or 'tensor flow' in x.lower() else 0)
```


```python
regex_df.jd_tensor.value_counts()
```




    0    670
    1     72
    Name: jd_tensor, dtype: int64




```python
regex_df['jd_jupyter'] = regex_df['Job Description'].apply(lambda x: 1 if 'jupyter' in x.lower() or 'notebook' in x.lower() else 0)
```


```python
regex_df.jd_jupyter.value_counts()
```




    0    697
    1     45
    Name: jd_jupyter, dtype: int64




```python
regex_df['jd_git'] = regex_df['Job Description'].apply(lambda x: 1 if 'git' in x.lower() or 'github' in x.lower() else 0)
```


```python
regex_df.jd_git.value_counts()
```




    0    581
    1    161
    Name: jd_git, dtype: int64




```python
regex_df['jd_library'] = regex_df['Job Description'].apply(lambda x: 1 if 'pandas' in x.lower() or 'numpy' in x.lower() or 'scikit' in x.lower() or 'scikit-learn' in x.lower() else 0)
```


```python
regex_df.jd_library.value_counts()
```




    0    672
    1     70
    Name: jd_library, dtype: int64




```python
regex_df['jd_aws'] = regex_df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
```


```python
regex_df.jd_aws.value_counts()
```




    0    566
    1    176
    Name: jd_aws, dtype: int64




```python
regex_df['jd_cloud'] = regex_df['Job Description'].apply(lambda x: 1 if 'cloud' in x.lower() or 'google cloud' in x.lower() else 0)
```


```python
regex_df.jd_cloud.value_counts()
```




    0    576
    1    166
    Name: jd_cloud, dtype: int64




```python
regex_df['jd_linux'] = regex_df['Job Description'].apply(lambda x: 1 if 'linux' in x.lower() else 0)
```


```python
regex_df.jd_linux.value_counts()
```




    0    686
    1     56
    Name: jd_linux, dtype: int64




```python
regex_df = regex_df.drop(['Unnamed: 0'], axis = 1)
```

# Job Title Parse


```python
import regex as re
```


```python
clean_title = []
for x in regex_df['Job Title']:
    match = re.findall(r'\bdata scientist\b', x, flags = re.IGNORECASE)
    match_2 = re.findall(r'\bdata science\b', x, flags = re.IGNORECASE)
    match_3 = re.findall(r'\bdata analyst\b', x, flags = re.IGNORECASE)
    match_4 = re.findall(r'\bdata analytics\b', x, flags = re.IGNORECASE)
    match_5 = re.findall(r'\bdata engineer\b', x, flags = re.IGNORECASE)
    match_6 = re.findall(r'\bresearch scientist\b', x, flags = re.IGNORECASE)
    match_7 = re.findall(r'\bmachine learning\b', x, flags = re.IGNORECASE)
    match_8 = re.findall(r'\bdirector\b', x, flags = re.IGNORECASE)
    match_9 = re.findall(r'\bmanager\b', x, flags = re.IGNORECASE)
    match_10 = re.findall(r'\bconsultant\b', x, flags = re.IGNORECASE)
    match_11 = re.findall(r'\bassociate director\b', x, flags = re.IGNORECASE)
    match_12 = re.findall(r'\banalyst\b', x, flags = re.IGNORECASE)
    match_13 = re.findall(r'\bdata engineering\b', x, flags = re.IGNORECASE)
    if match or match_2:
        clean_title.append(x.replace(x, 'Data Scientist'))
        #regex_df[job_title] = job_title.append(x.replace(x, 'Data Scientist'))
    elif match_3:
        clean_title.append(x.replace(x, 'Data Analyst'))
    elif match_4:
        clean_title.append(x.replace(x, 'Data Analyst'))
    elif match_5:
        clean_title.append(x.replace(x, 'Data Engineer'))
    elif match_6:
        clean_title.append(x.replace(x, 'Research Scientist'))
    elif match_7:
        clean_title.append(x.replace(x, 'Machine Learning Engineer'))
    elif match_8:
        clean_title.append(x.replace(x, 'Director'))
    elif match_9:
        clean_title.append(x.replace(x, 'Manager'))
    elif match_10:
        clean_title.append(x.replace(x, 'Consultant'))
    elif match_11:
        clean_title.append(x.replace(x, 'Associate Director'))
    elif match_12:
        clean_title.append(x.replace(x, 'Data Analyst'))
    elif match_13:
        clean_title.append(x.replace(x, 'Data Engineer'))
    else:
        clean_title.append(x.replace(x, 'NA'))
```


```python
regex_df['clean_title'] = clean_title
```


```python
regex_df.clean_title.value_counts()
```




    Data Scientist               313
    NA                           143
    Data Engineer                115
    Data Analyst                 106
    Research Scientist            22
    Machine Learning Engineer     15
    Manager                       13
    Director                       8
    Consultant                     7
    Name: clean_title, dtype: int64




```python
regex_df.clean_title.value_counts().sum()
```




    742




```python
regex_df.clean_title.value_counts().to_csv('Clean_titles.csv', index = False)
```

# JD Length


```python
regex_df['jd_length'] = regex_df['Job Description'].apply(lambda x: len(x))
```

# Competitor Count


```python
regex_df['comp_count'] = regex_df['Competitors'].apply(lambda x: len(x.split(',')) if x != '-1' else 0)
```

# Per hour to annual



```python
regex_df['min salary'] = pd.to_numeric(regex_df['min salary'], errors='coerce')
regex_df['max_salary'] = pd.to_numeric(regex_df['max_salary'], errors='coerce')
```


```python
regex_df['min salary'] = regex_df.apply(lambda x: round(x['min salary']*1.92,0) if x.per_hour == 1 else x['min salary'], axis =1)
```


```python
regex_df['max_salary'] = regex_df.apply(lambda x: round(x['max_salary']*1.92,0) if x.per_hour == 1 else x['max_salary'], axis =1)
```


```python
regex_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Job Title</th>
      <th>Salary Estimate</th>
      <th>Job Description</th>
      <th>Rating</th>
      <th>Company Name</th>
      <th>Location</th>
      <th>Headquarters</th>
      <th>Size</th>
      <th>Founded</th>
      <th>Type of ownership</th>
      <th>Industry</th>
      <th>Sector</th>
      <th>Revenue</th>
      <th>Competitors</th>
      <th>per_hour</th>
      <th>employer_provided</th>
      <th>min salary</th>
      <th>max_salary</th>
      <th>company_txt</th>
      <th>State</th>
      <th>City</th>
      <th>same_state_hq</th>
      <th>company_age</th>
      <th>jd_python</th>
      <th>jd_sql</th>
      <th>jd_excel</th>
      <th>jd_spark</th>
      <th>jd_apache</th>
      <th>jd_r</th>
      <th>jd_tensor</th>
      <th>jd_jupyter</th>
      <th>jd_git</th>
      <th>jd_library</th>
      <th>jd_aws</th>
      <th>jd_cloud</th>
      <th>jd_linux</th>
      <th>clean_title</th>
      <th>jd_length</th>
      <th>comp_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Data Scientist</td>
      <td>$53K-$91K (Glassdoor est.)</td>
      <td>Data Scientist\nLocation: Albuquerque, NM\nEdu...</td>
      <td>3.8</td>
      <td>Tecolote Research\n3.8</td>
      <td>Albuquerque, NM</td>
      <td>Goleta, CA</td>
      <td>501 to 1000 employees</td>
      <td>1973</td>
      <td>Company - Private</td>
      <td>Aerospace &amp; Defense</td>
      <td>Aerospace &amp; Defense</td>
      <td>$50 to $100 million (USD)</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>53.0</td>
      <td>91.0</td>
      <td>Tecolote Research</td>
      <td>NM</td>
      <td>Albuquerque</td>
      <td>0</td>
      <td>50</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Data Scientist</td>
      <td>2536</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Healthcare Data Scientist</td>
      <td>$63K-$112K (Glassdoor est.)</td>
      <td>What You Will Do:\n\nI. General Summary\n\nThe...</td>
      <td>3.4</td>
      <td>University of Maryland Medical System\n3.4</td>
      <td>Linthicum, MD</td>
      <td>Baltimore, MD</td>
      <td>10000+ employees</td>
      <td>1984</td>
      <td>Other Organization</td>
      <td>Health Care Services &amp; Hospitals</td>
      <td>Health Care</td>
      <td>$2 to $5 billion (USD)</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>63.0</td>
      <td>112.0</td>
      <td>University of Maryland Medical System</td>
      <td>MD</td>
      <td>Linthicum</td>
      <td>1</td>
      <td>39</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Data Scientist</td>
      <td>4783</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Data Scientist</td>
      <td>$80K-$90K (Glassdoor est.)</td>
      <td>KnowBe4, Inc. is a high growth information sec...</td>
      <td>4.8</td>
      <td>KnowBe4\n4.8</td>
      <td>Clearwater, FL</td>
      <td>Clearwater, FL</td>
      <td>501 to 1000 employees</td>
      <td>2010</td>
      <td>Company - Private</td>
      <td>Security Services</td>
      <td>Business Services</td>
      <td>$100 to $500 million (USD)</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>80.0</td>
      <td>90.0</td>
      <td>KnowBe4</td>
      <td>FL</td>
      <td>Clearwater</td>
      <td>1</td>
      <td>13</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Data Scientist</td>
      <td>3461</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Data Scientist</td>
      <td>$56K-$97K (Glassdoor est.)</td>
      <td>*Organization and Job ID**\nJob ID: 310709\n\n...</td>
      <td>3.8</td>
      <td>PNNL\n3.8</td>
      <td>Richland, WA</td>
      <td>Richland, WA</td>
      <td>1001 to 5000 employees</td>
      <td>1965</td>
      <td>Government</td>
      <td>Energy</td>
      <td>Oil, Gas, Energy &amp; Utilities</td>
      <td>$500 million to $1 billion (USD)</td>
      <td>Oak Ridge National Laboratory, National Renewa...</td>
      <td>0</td>
      <td>0</td>
      <td>56.0</td>
      <td>97.0</td>
      <td>PNNL</td>
      <td>WA</td>
      <td>Richland</td>
      <td>1</td>
      <td>58</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Data Scientist</td>
      <td>3883</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Data Scientist</td>
      <td>$86K-$143K (Glassdoor est.)</td>
      <td>Data Scientist\nAffinity Solutions / Marketing...</td>
      <td>2.9</td>
      <td>Affinity Solutions\n2.9</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>51 to 200 employees</td>
      <td>1998</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>Unknown / Non-Applicable</td>
      <td>Commerce Signals, Cardlytics, Yodlee</td>
      <td>0</td>
      <td>0</td>
      <td>86.0</td>
      <td>143.0</td>
      <td>Affinity Solutions</td>
      <td>NY</td>
      <td>New York</td>
      <td>1</td>
      <td>25</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Data Scientist</td>
      <td>2728</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>950</th>
      <td>Sr Scientist, Immuno-Oncology - Oncology</td>
      <td>$58K-$111K (Glassdoor est.)</td>
      <td>Site Name: USA - Massachusetts - Cambridge\nPo...</td>
      <td>3.9</td>
      <td>GSK\n3.9</td>
      <td>Cambridge, MA</td>
      <td>Brentford, United Kingdom</td>
      <td>10000+ employees</td>
      <td>1830</td>
      <td>Company - Public</td>
      <td>Biotech &amp; Pharmaceuticals</td>
      <td>Biotech &amp; Pharmaceuticals</td>
      <td>$10+ billion (USD)</td>
      <td>Pfizer, AstraZeneca, Merck</td>
      <td>0</td>
      <td>0</td>
      <td>58.0</td>
      <td>111.0</td>
      <td>GSK</td>
      <td>MA</td>
      <td>Cambridge</td>
      <td>0</td>
      <td>193</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NA</td>
      <td>6162</td>
      <td>3</td>
    </tr>
    <tr>
      <th>951</th>
      <td>Senior Data Engineer</td>
      <td>$72K-$133K (Glassdoor est.)</td>
      <td>THE CHALLENGE\nEventbrite has a world-class da...</td>
      <td>4.4</td>
      <td>Eventbrite\n4.4</td>
      <td>Nashville, TN</td>
      <td>San Francisco, CA</td>
      <td>1001 to 5000 employees</td>
      <td>2006</td>
      <td>Company - Public</td>
      <td>Internet</td>
      <td>Information Technology</td>
      <td>$100 to $500 million (USD)</td>
      <td>See Tickets, TicketWeb, Vendini</td>
      <td>0</td>
      <td>0</td>
      <td>72.0</td>
      <td>133.0</td>
      <td>Eventbrite</td>
      <td>TN</td>
      <td>Nashville</td>
      <td>0</td>
      <td>17</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Data Engineer</td>
      <td>6130</td>
      <td>3</td>
    </tr>
    <tr>
      <th>952</th>
      <td>Project Scientist - Auton Lab, Robotics Institute</td>
      <td>$56K-$91K (Glassdoor est.)</td>
      <td>The Auton Lab at Carnegie Mellon University is...</td>
      <td>2.6</td>
      <td>Software Engineering Institute\n2.6</td>
      <td>Pittsburgh, PA</td>
      <td>Pittsburgh, PA</td>
      <td>501 to 1000 employees</td>
      <td>1984</td>
      <td>College / University</td>
      <td>Colleges &amp; Universities</td>
      <td>Education</td>
      <td>Unknown / Non-Applicable</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>56.0</td>
      <td>91.0</td>
      <td>Software Engineering Institute</td>
      <td>PA</td>
      <td>Pittsburgh</td>
      <td>1</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NA</td>
      <td>3078</td>
      <td>0</td>
    </tr>
    <tr>
      <th>953</th>
      <td>Data Science Manager</td>
      <td>$95K-$160K (Glassdoor est.)</td>
      <td>Data Science ManagerResponsibilities:\n\nOvers...</td>
      <td>3.2</td>
      <td>Numeric, LLC\n3.2</td>
      <td>Allentown, PA</td>
      <td>Chadds Ford, PA</td>
      <td>1 to 50 employees</td>
      <td>-1</td>
      <td>Company - Private</td>
      <td>Staffing &amp; Outsourcing</td>
      <td>Business Services</td>
      <td>$5 to $10 million (USD)</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>95.0</td>
      <td>160.0</td>
      <td>Numeric</td>
      <td>PA</td>
      <td>Allentown</td>
      <td>1</td>
      <td>NAN</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Data Scientist</td>
      <td>1642</td>
      <td>0</td>
    </tr>
    <tr>
      <th>955</th>
      <td>Research Scientist – Security and Privacy</td>
      <td>$61K-$126K (Glassdoor est.)</td>
      <td>Returning Candidate? Log back in to the Career...</td>
      <td>3.6</td>
      <td>Riverside Research Institute\n3.6</td>
      <td>Beavercreek, OH</td>
      <td>Arlington, VA</td>
      <td>501 to 1000 employees</td>
      <td>1967</td>
      <td>Nonprofit Organization</td>
      <td>Federal Agencies</td>
      <td>Government</td>
      <td>$50 to $100 million (USD)</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>61.0</td>
      <td>126.0</td>
      <td>Riverside Research Institute</td>
      <td>OH</td>
      <td>Beavercreek</td>
      <td>0</td>
      <td>56</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Research Scientist</td>
      <td>3673</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>742 rows × 39 columns</p>
</div>




```python

```


```python

```


```python

```


```python

```


```python

```
