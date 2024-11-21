# Simple Linear Regression 
A straightforward  approcah for predicting a quantitative response Y on the basis of a single predictor Variable X. 
It assumes that there is a approximately a linear relationship between X and Y. 
![image](https://github.com/user-attachments/assets/3c251531-911a-4b9c-8b61-6ddd732a780b)
 SO, we are regressing Sales onto TV and beta0 and beta 1 are two unknown constants that
 represent the intercept and the slope terms in the linear model. 



# KNN
 ![image](https://github.com/user-attachments/assets/ef8149fc-59f5-46b7-95bd-d4974cb8ed73)
![image](https://github.com/user-attachments/assets/369e778c-04e0-46f3-8209-27d858fbfb17)











Instructions:
Read the Advertisement data and view the top rows of the dataframe to get an understanding of the data and the columns.

Select the first 7 observations and the columns TV and sales  to make a new data frame.

Create a scatter plot of the new data frame TV budget vs sales .


Hints: 
The following are direct links to documentation for some of the functions used in this exercise. As always, if you are unsure how to use a function, refer to its documentation. 

pd.read_csv(filename)
 Returns a pandas dataframe containing the data and labels from the file data


df.iloc[]
Returns a subset of the dataframe that is contained in the row range passed as the argument


df.head()
Returns the first 5 rows of the dataframe with the column names


plt.scatter()
A scatter plot of y vs. x with varying marker size and/or color


plt.xlabel()
This is used to specify the text to be displayed as the label for the x-axis


plt.ylabel()
This is used to specify the text to be displayed as the label for the y-axis


plt.title()
This is used to specify the title to be displayed for the plot 




## Instead of just plotting seven points, experiment to plot all points.
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
# "Advertising.csv" containts the data set used in this exercise
data_filename = 'Advertising.csv'

# Read the file "Advertising.csv" file using the pandas library
df = pd.read_csv(data_filename)
df
# Use a scatter plot for plotting a graph of TV vs Sales
plt.scatter(df['TV'],df['Sales'])

# Add axis labels for clarity (x : TV budget, y : Sales)
plt.xlabel("TV budget")
plt.ylabel("Sales")

# Add plot title 
plt.title("data frame TV budget vs sales ALL data")
