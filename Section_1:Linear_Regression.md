# Pre-Reading Questions with Answers: 
1.The authors in the book call the error term "a catch-all". Which of the following is not part of the catch-all error? A mistake in the code
2.Sample and Population
The sample mean and population means are different. True

## Introduction to Regression - Part 1. KNN



The advertising data set (below) consists of the sales of a particular product in 200 different markets, and advertising budgets for the product in each of those markets for three different media:
TV, radio, and newspaper. We have 4 columns and 200 rows where budgets are given in units of $1,000 and sales in 1,000 sales.
Variables whose values we use to make our prediction. These are known as predictors, features, or covariates.
Variables whose values we want to predict. These are known as outcomes, response variables, or dependent variables.
![image](https://github.com/user-attachments/assets/527c33b4-6c77-42a3-ae49-039f74c68a0b)

![image](https://github.com/user-attachments/assets/5ba7eee2-95a4-4ae6-ad68-54d8d5472b59)

![image](https://github.com/user-attachments/assets/dbfb7e17-9fdc-4d02-b133-9805890a8f0c)




The aim of this exercise is to plot TV Ads vs Sales based on the Advertisement dataset:
### Instructions:
Read the Advertisement data and view the top rows of the dataframe to get an understanding of the data and the columns.
Select the first 7 observations and the columns TV and sales  to make a new data frame.
Create a scatter plot of the new data frame TV budget vs sales .
____________________________________________________________________________________________________________________________

pd.read_csv(filename): Returns a pandas dataframe containing the data and labels from the file data

df.iloc[] :Returns a subset of the dataframe that is contained in the row range passed as the argument

df.head():Returns the first 5 rows of the dataframe with the column names

plt.scatter():A scatter plot of y vs. x with varying marker size and/or color

plt.xlabel():This is used to specify the text to be displayed as the label for the x-axis

plt.ylabel():This is used to specify the text to be displayed as the label for the y-axis

plt.title():This is used to specify the title to be displayed for the plot 
______________________________________________________________________________________________________________________________
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

#"Advertising.csv" containts the data set used in this exercise
data_filename = 'Advertising.csv'

#Read the file "Advertising.csv" file using the pandas library
df = pd.read_csv(data_filename)

#Get a quick look of the data
df.head()


###edTest(test_pandas) ###
#Create a new dataframe by selecting the first 7 rows of the current dataframe
df_new = df.head(7)


#Print your new dataframe to see if you have selected 7 rows correctly
print(df_new)

#Use a scatter plot for plotting a graph of TV vs Sales
plt.scatter(df_new['TV'],df_new['Sales'])

#Add axis labels for clarity (x : TV budget, y : Sales)
plt.xlabel("TV budget")
plt.ylabel("Sales")

#Add plot title 
plt.title("data frame TV budget vs sales")


### Post-Exercise Question: Instead of just plotting seven points, experiment to plot all points.
#Your code here
#Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
#"Advertising.csv" containts the data set used in this exercise
data_filename = 'Advertising.csv'

#Read the file "Advertising.csv" file using the pandas library
df = pd.read_csv(data_filename)
df
#Use a scatter plot for plotting a graph of TV vs Sales
plt.scatter(df['TV'],df['Sales'])

#Add axis labels for clarity (x : TV budget, y : Sales)
plt.xlabel("TV budget")
plt.ylabel("Sales")

#Add plot title 
plt.title("data frame TV budget vs sales ALL data")
![image](https://github.com/user-attachments/assets/11f0f0b2-381e-426b-bc04-b638906dc75d)

