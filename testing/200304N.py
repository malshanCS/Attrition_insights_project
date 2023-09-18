import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
chatterbox = pd.read_csv('employee.csv')
employee = pd.read_csv('employee.csv')
# employee.head(20)
### Change Genders and Maritial Status Accordingly
# If the Gender is Male, Change title to Mr
employee.loc[employee['Gender'] == 'Male', ['Title']] = 'Mr'


# If the Gender is Female, and Status is Single, Change title to Miss
employee.loc[(employee['Gender'] == 'Female') & (employee['Marital_Status'] == 'Single'), 'Title'] = 'Miss'

# If the Gender is Female, and Status is Married, Change title to Miss
employee.loc[(employee['Gender'] == 'Female') & (employee['Marital_Status'] == 'Married'), 'Title'] = 'Ms'



# # If name contains Ms, Change gender to Female and title to Ms
# employee.loc[employee['Name'].str.contains('Ms '), ['Title','Gender']] = ['Ms','Female']


# # If name contains Miss, Change gender to Female and title to Miss
# employee.loc[employee['Name'].str.contains('Miss '), ['Title','Gender']] = ['Miss','Female']


# # First two are Females and Last one is a Male
# employee.loc[employee['Employee_No'].isin([551,601]), ['Title']] = 'Ms'


# employee.loc[employee['Employee_No']== 2369 , ['Gender']] = 'Male'


# employee.loc[employee['Employee_No'].isin([1886,2509]), ['Title']] = 'Mr'


# employee.loc[(employee['Title'] == 'Ms') & (employee['Gender'] == 'Male'),['Gender']]='Female'


# employee.loc[(employee['Title'].isin(['Ms','Mrs','Miss'])) & (employee['Gender'] == 'Male'), ['Gender']] = 'Female'

# employee.tail(20)

### Some Extra works
employee.loc[employee['Year_of_Birth'] == "'0000'"]
employee.loc[employee['Year_of_Birth'] == "'0000'", 'Year_of_Birth'] = '0000'
employee['Date_Joined'] = pd.to_datetime(employee['Date_Joined'])
# employee.loc[employee['Year_of_Birth']=='0000']
employee_copy = employee.copy()
# Extract the year from the 'Date_Joined' column
employee['Joining_Year'] = employee['Date_Joined'].dt.year # Extract the year from the 'Date_Joined' column
# employee.isna().sum()   
# # drop rows where year of birth is 0000
# employee = employee[employee['Year_of_Birth'] != '0000']
employee['Year_of_Birth'] = employee['Year_of_Birth'].astype(int)
# set 0000 to NaN in Year_of_Birth
employee.loc[employee['Year_of_Birth'] == 0, 'Year_of_Birth'] = np.nan
# employee.loc[(employee['Date_Resigned'] != '0000-00-00') | (employee['Date_Resigned'] != '\\N'), 'Status'] = 'Inactive'
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

employee['Gender_temp'] = label_encoder.fit_transform(employee['Gender'].astype(str))
employee['Status_temp'] = label_encoder.fit_transform(employee['Status'].astype(str))
employee['Title_temp'] = label_encoder.fit_transform(employee['Title'].astype(str))


employee['Employment_Category_temp'] = label_encoder.fit_transform(employee['Employment_Category'].astype(str))
employee['Employment_Type_temp'] = label_encoder.fit_transform(employee['Employment_Type'].astype(str))
#

# # Create a new column called age
# employee['Age'] = 2023 - employee['Year_of_Birth']
# employee.isna().sum()
# employee
employee['Age_at_Joining'] = employee['Joining_Year'] - employee['Year_of_Birth']
employee['Years_of_Service'] = 2023 - employee['Joining_Year']

test_df = employee[pd.isnull(employee['Year_of_Birth'])]
train_df = employee[pd.notnull(employee['Year_of_Birth'])]
train_df = train_df.drop(columns=['Name','Title','Marital_Status','Gender','Date_Resigned','Status','Date_Joined','Inactive_Date','Reporting_emp_1','Reporting_emp_2','Employment_Category','Employment_Type','Religion','Designation'])

test_df = test_df.drop(columns=['Name','Title','Gender','Marital_Status','Date_Resigned','Status','Date_Joined','Inactive_Date','Reporting_emp_1','Reporting_emp_2','Employment_Category','Employment_Type','Religion','Designation'])
# train_df['Age_at_Joining'] = train_df['Joining_Year'] - train_df['Year_of_Birth']
# train_df['Years_of_Service'] = 2023 - train_df['Joining_Year']

# Drop unnecessary columns
train_df.drop(['Joining_Year'], axis=1, inplace=True)
# train_df
# # test_df['Age_at_Joining'] = test_df['Joining_Year'] - test_df['Year_of_Birth']
# test_df['Years_of_Service'] = 2023 - test_df['Joining_Year']

# Drop unnecessary columns
test_df.drop(['Joining_Year'], axis=1, inplace=True)
# import train_test_split
from sklearn.model_selection import train_test_split

# Split the train dataset into train_X, train_y, val_X, val_y
train_X, val_X, train_y, val_y = train_test_split(train_df.drop(columns=['Year_of_Birth']), train_df['Year_of_Birth'], test_size=0.2, random_state=42) 
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVC

# # Separate the features and target variables in the train_df
# train_features = train_df.drop(['Year_of_Birth'], axis=1)
# train_target = train_df['Year_of_Birth']

# # Separate the features in the test_df
# test_features = test_df.drop(['Year_of_Birth'], axis=1)


# Initialize the Random Forest regression model
rf_model = RandomForestRegressor()

# Train the model on the train_df
rf_model.fit(train_X, train_y)

# train_df.isna().sum()
# train svc model
svc_model = SVC()
svc_model.fit(train_X, train_y)
# test the accuracy of the model
# rf_model.score(val_X, val_y)
# accuracy of svc model
# svc_model.score(val_X, val_y)
# test_df.isna().sum()    
# fill Age_at_Joining with mean
test_df['Age_at_Joining'].fillna(train_df['Age_at_Joining'].mean(), inplace=True)
# test_df.isna().sum()

# Predict the missing values in the test_df using the trained model
imputed_values = rf_model.predict(test_df.drop(columns=['Year_of_Birth']))
# imputed_values



# Update the 'Year_of_Birth' column in the test_df with the imputed values
test_df['Year_of_Birth'] = imputed_values
# Concatenate the train_df and test_df
imp = pd.concat([train_df, test_df], ignore_index=True)
# imp
# convert the 'Year_of_Birth' column to integer type
imp['Year_of_Birth'] = imp['Year_of_Birth'].astype(int)
# imp
employee_copy = employee.copy()
employee = employee_copy.copy()
# replace the 'Year_of_Birth' column in the original employee dataframe with the imputed values based on Employee_No
employee_copy['Year_of_Birth'] = employee_copy['Employee_No'].map(imp.set_index('Employee_No')['Year_of_Birth'])

# employee_copy
# convert date joined to datetime
employee_copy['Date_Joined'] = pd.to_datetime(employee_copy['Date_Joined'])
# employee_copy.isna().sum()
# get age at joining
employee_copy['Age_at_Joining'] = employee_copy['Date_Joined'].dt.year - employee_copy['Year_of_Birth']
employee_copy['Age_at_Joining'] = employee_copy['Age_at_Joining'].astype(int)
# employee_copy['Age_at_Joining'].describe()
# # drop rows where age at joining is less than 18
# employee = employee[employee['Age_at_Joining'] >= 18]
# get date_resigned = "\\N" to 0000-00-00
# employee.loc[employee['Date_Resigned'] == '\\N', 'Date_Resigned'] = '0000-00-00'

# employee['Employment_Type']


# employee_copy
### Data imputations in Marital_Status column
employee_copy_copy = employee.copy()
employee = employee_copy.copy()
from sklearn.ensemble import RandomForestClassifier
# employee.columns
### Imputing Missing Values in year of birth
# employee.columns.to_list()
# employee.isna().sum()
# Assuming 'df' is the DataFrame containing the dataset
test_df = employee[pd.isnull(employee['Marital_Status'])]
train_df = employee[pd.notnull(employee['Marital_Status'])]
train_df = train_df.drop(columns=['Name','Title','Gender','Date_Resigned','Status','Date_Joined','Inactive_Date','Reporting_emp_1','Reporting_emp_2','Employment_Category','Employment_Type','Religion','Designation'])

test_df = test_df.drop(columns=['Name','Title','Gender','Date_Resigned','Status','Date_Joined','Inactive_Date','Reporting_emp_1','Reporting_emp_2','Employment_Category','Employment_Type','Religion','Designation'])
train_df['Marital_Status'] = label_encoder.fit_transform(train_df['Marital_Status'].astype(str))
# train_df
from sklearn.model_selection import train_test_split
X = train_df.drop('Marital_Status', axis=1)
y = train_df['Marital_Status']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
# use X_val and y_val to evaluate the model
y_pred = model.predict(X_val)

from sklearn.metrics import accuracy_score
# print('Accuracy Score:', accuracy_score(y_val, y_pred))


# test_df['Marital_Status'] = label_encoder.transform(test_df['Marital_Status'].astype(str))
test_df['Marital_Status'] = model.predict(test_df.drop('Marital_Status', axis=1))

# test_df['Marital_Status']
# concat train_df and test_df into employee_after df
employee_after = pd.concat([train_df, test_df], axis=0)
# employee 
# employee_after
emmployee_copy_copy = employee.copy()

# replace marital status in employee with marital status in employee_after based on employee_no
employee = pd.merge(employee, employee_after[['Employee_No', 'Marital_Status']], on='Employee_No', how='left')
# employee
emmployee_copy_copy = employee.copy()
# Assuming 'df' is the DataFrame containing the data
dick = {employee.loc[employee['Employee_No'] == 347, 'Marital_Status_y'].values[0]: 'Married', employee.loc[employee['Employee_No'] == 2836, 'Marital_Status_y'].values[0]: 'Single'}
employee['Marital_Status_y'] = employee['Marital_Status_y'].replace(dick)
# employee
# employee.isna().sum()
employee.drop(columns=['Marital_Status_x'], inplace=True)
# rename Marital_Status_y column to Marital_Status
employee = employee.rename(columns={'Marital_Status_y': 'Marital_Status'})
# employee
employee_original = pd.read_csv('employee.csv')
# employee_original
cols = employee_original.columns.to_list()
# cols
employee_cleaned = employee[cols]
# employee_cleaned.isna().sum()

# If the Gender is Female, and Status is Married, Change title to Miss
employee_cleaned.loc[(employee_cleaned['Gender'] == 'Female') & (employee_cleaned['Marital_Status'] == 'Married'), 'Title'] = 'Ms'


# If the Gender is Female, and Status is Single, Change title to Miss
employee_cleaned.loc[(employee_cleaned['Gender'] == 'Female') & (employee_cleaned['Marital_Status'] == 'Single'), 'Title'] = 'Miss'

## Doing too much
# export employee_cleaned as csv

employee_cleaned.to_csv('employee_preprocess_200304N.csv', index=False)