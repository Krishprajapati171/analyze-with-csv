import pandas as  pd
import numpy as np
import matplotlib.pyplot as pt
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import rotate_axes


#######################################INTRODUCTION TO EMPLOYEE PERFORMANCE ANALYSIS ###################################

introduction_text = """Employee Performance Analysis - Introduction

In any organization, analyzing employee performance is crucial for workforce management, salary structuring, 
and career growth planning. This project aims to evaluate employee performance using real-world data, focusing 
on key metrics such as salary distribution, performance scores, promotions, and experience correlations.

Using the provided dataset of 5,001 employees, we perform an in-depth analysis of:
1. Salary Trends Across Departments – Identifying which departments have the highest and lowest salaries.
2. Performance Score Distribution – Understanding employee productivity levels.
3. Promotion Trends – Examining how performance affects career growth.
4. Experience vs. Salary Relationship – Assessing whether tenure significantly impacts salary increments.

Objective:
The goal of this analysis is to help HR and management teams make data-driven decisions about compensation, 
employee retention, and workforce development strategies.
"""

########################################## LOAD AND EXPLORATION OF THE DATA-SET########################################

# Load and Explore the Data
df=pd.read_csv("large_employee_data.csv")
print(df.head())
print(df.info())


# : Data Cleaning
print(df.drop_duplicates())
print(df.fillna('new'))


# Convert Data Types

print(df['Years at Company'].astype(float))
print(df['Salary'].astype(float))


# ################################################Data Analysis#########################################################


#  Average Salary by Department
print(df.groupby('Department')['Salary'].mean().sort_values())



#  Performance Score Distribution

print(df['Performance Score'].value_counts())


# Promotion Rate Analysis

print(df['Promotion Status'].value_counts())



############################################## Data Visualization#######################################################

# 5.4 Years at Company vs. Salary Growth

print(df.groupby("Years at Company")["Salary"].mean().plot(kind="line", title="Salary Growth Over Time"))
pt.xticks(rotation=90)
pt.title("Average Salary by Department")
pt.show()


# 6.2 Performance Score Distribution
sns.histplot(df['Performance Score'],bins=5)
pt.title("Performance Score Distribution")
pt.show()



# 6.3 Promotion Rate Pie Chart

print(df['Promotion Status'].value_counts().plot(kind='pie',title='Promotion Rate'))
pt.show()



# 6.4 Salary Growth Over Experience

sns.lineplot(x=df['Years at Company'],y=df['Salary'])
pt.title("Salary Growth Over  Experience")
pt.xlabel("Years at Company")
pt.ylabel('Average Salary')
pt.show()



#6.5 Highest Salary in the Department

print(df.groupby('Department')['Salary'].mean().sort_values().plot(kind='bar',title='Almost Same for the All department'))
pt.xlabel('Department')
pt.ylabel('Salary')
pt.xticks(rotation=90)
pt.show()




#################################################### Conclusion#########################################################




conclusion_text = """Employee Performance Analysis - Conclusion

1. Salary Trends:
   - HR has the lowest average salary ($94,625.95), while Marketing has the highest ($96,079.97).
   - Overall, salary distribution across departments is balanced.

2. Performance Score Distribution:
   - The average performance score is 3.77.
   - 25% of employees score above 4.4 (top performers).
   - Few employees have a score below 3.0, indicating training needs.

3. Promotion Trends:
   - 49.28% of employees were promoted.
   - Higher performance scores increase promotion chances.

4. Experience vs. Salary Correlation:
   - Weak correlation (0.0093) between years at the company and salary.
   - Other factors (performance, department) play a bigger role in salary growth.

Recommendations:
- Adjust HR salaries for fair compensation.
- Implement performance-based salary increments.
- Provide training programs for low performers.
- Ensure top performers receive recognition even if not promoted.

"""

# Save to a text file
with open("Employee_Performance_Analysis_Conclusion.txt", "w") as file:
    file.write(conclusion_text)

print("Conclusion saved as 'Employee_Performance_Analysis_Conclusion.txt'")
