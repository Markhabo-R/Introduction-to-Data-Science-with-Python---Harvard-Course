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




##The aim of this exercise is to plot TV Ads vs Sales based on the Advertisement dataset:
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
______________________________________________________________________________________________________________________
## kNN Regression
np.argsort():Returns the indices that would sort an array. 


df.iloc[]:Returns a subset of the dataframe that is contained in the column range passed as the argument.


plt.plot( ):Plot y versus x as lines and/or markers.


df.values:Returns a Numpy representation of the DataFrame.


pd.idxmin():Returns index of the first occurrence of minimum over requested axis.


np.min():Returns the minimum along a given axis.


np.max():Returns the maximum along a given axis.


model.fit( ):Fit the k-nearest neighbors regressor from the training dataset.


model.predict( ):Predict the target for the provided data.


np.zeros():Returns a new array of given shape and type, filled with zeros.


train_test_split(X,y):Split arrays or matrices into random train and test subsets. 


np.linspace():Returns evenly spaced numbers over a specified interval.


KNeighborsRegressor(n_neighbors=k_value):Regression-based on k-nearest neighbors. 


### Instructions:Part 1: KNN by hand for k=1
Read the Advertisement data.
Get a subset of the data from row 5 to row 13.
Apply the kNN algorithm by hand and plot the graph 

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
%matplotlib inline

#Read the data from the file "Advertising.csv"
filename = 'Advertising.csv'
df_adv = pd.read_csv(filename)

#Take a quick look of the dataset
df_adv.head()

#### Part 1: KNN by hand for k=1
#Get a subset of the data i.e. rows 5 to 13:  Use the TV column as the predictor
x_true = df_adv.TV.iloc[5:13]



#Use the Sales column as the response
y_true = df_adv.Sales.iloc[5:13]

#Sort the data to get indices ordered from lowest to highest TV values
idx = np.argsort(x_true).values 

#Get the predictor data in the order given by idx above
x_true  = x_true.iloc[idx].values

#Get the response data in the order given by idx above
y_true  = y_true.iloc[idx].values


###edTest(test_findnearest) ###
#Define a function that finds the index of the nearest neighbor and returns the value of the nearest neighbor.  
#Note that this is just for k = 1 where the distance function is simply the absolute value.
def find_nearest(array,value):

def find_nearest(array,value):
    
    #Hint: To find idx, use .idxmin() function on the series
    idx = pd.Series(np.abs(array-value)).idxmin()

    #Return the nearest neighbor index and value
    return idx, array[idx]

#Create some synthetic x-values (might not be in the actual dataset)
x = np.linspace(np.min(x_true), np.max(x_true))

#Initialize the y-values for the length of the synthetic x-values to zero
y = np.zeros((len(x)))

#Apply the KNN algorithm to predict the y-value for the given x value
for i, xi in enumerate(x):

    # Get the Sales values closest to the given x value
    y[i] = y_true[find_nearest(x_true, xi )[0]]


#Plot the synthetic data along with the predictions    
plt.plot(x, y, '-.')

#Plot the original data using black x's.
plt.plot(x_true, y_true, 'kx')

#Set the title and axis labels
plt.title('TV vs Sales')
plt.xlabel('TV budget in $1000')
plt.ylabel('Sales in $1000')

![image](https://github.com/user-attachments/assets/f56a4e01-c88e-4e96-bbd3-53e4bcdf4cea)


### Part 2: Using sklearn package

Read the Advertisement dataset.

Split the data into train and test sets using the train_test_split() function.

Set k_list  as the possible k values ranging from 1 to 70.

For each value of k in k_list:

Use sklearn KNearestNeighbors() to fit train data.

Predict on the test data.

Use the helper code to get the second plot above for k=1,10,70.
____________________________________________________________________________________________________________
#Read the data from the file "Advertising.csv"
data_filename = 'Advertising.csv'
df = pd.read_csv(data_filename)

#Set 'TV' as the 'predictor variable'   
x = df['TV'].values.reshape(-1, 1)

#Set 'Sales' as the response variable 'y' 
y = df['Sales'].values



###edTest(test_shape) ###

#Split the dataset in training and testing with 60% training set and 40% testing set with random state = 42
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6,random_state=42)



###edTest(test_nums) ###

#Choose the minimum k value based on the instructions given above
k_value_min = 1

#Choose the maximum k value based on the instructions given above
k_value_max = 70


#Create a list of integer k values between k_value_min and k_value_max using linspace
k_list = np.linspace(k_value_min, k_value_max, 70)


#Set the grid to plot the values
fig, ax = plt.subplots(figsize=(10,6))

#Variable used to alter the linewidth of each plot
j=0

#Loop over all the k values
for k_value in k_list:   
    
    # Creating a kNN Regression model 
    model = KNeighborsRegressor(n_neighbors=int(k_value))
    
    # Fitting the regression model on the training data 
    model.fit(x_train, y_train)
    
    # Use the trained model to predict on the test data 
    y_pred = model.predict(x_test)
    
    # Helper code to plot the data along with the model predictions
    colors = ['grey','r','b']
    if k_value in [1,10,70]:
        xvals = np.linspace(x.min(),x.max(),100).reshape(-1,1)
        ypreds = model.predict(xvals)
        ax.plot(xvals, ypreds,'-',label = f'k = {int(k_value)}',linewidth=j+2,color = colors[j])
        j+=1
        
ax.legend(loc='lower right',fontsize=20)
ax.plot(x_train, y_train,'x',label='train',color='k')
ax.set_xlabel('TV budget in $1000',fontsize=20)
ax.set_ylabel('Sales in $1000',fontsize=20)
plt.tight_layout()


![image](https://github.com/user-attachments/assets/0654e41e-1226-4260-afc3-ee11dae4b20f)

## Finding the Best k in kNN Regression
### The goal here is to find the value of k of the best performing model based on the test MSE.
____________________________________________________________________________________________________
train_test_split(X,y):Split arrays or matrices into random train and test subsets. 


np.linspace():Returns evenly spaced numbers over a specified interval.


KNeighborsRegressor(n_neighbors=k_value):Regression-based on k-nearest neighbors. 


model.predict():Predict the target for the provided data.


mean_squared_error():Computes the mean squared error regression loss.


dict.keys() :Returns a view object that displays a list of all the keys in the dictionary.


dict.values(): Returns a list of all the values available in a given dictionary.


plt.plot():Plot y versus x as lines and/or markers.


dict.items() :Returns a list of dict's (key, value) tuple pairs.
_____________________________________________________________________________________________
Instructions:
Read the data into a Pandas dataframe object. 

Select the sales column as the response variable and TV budget column as the predictor variable.

Make a train-test split using sklearn.model_selection.train_test_split .

Create a list of integer k values using numpy.linspace .

For each value of k

Fit a kNN regression on train set.

Calculate MSE on test set and store it.

Plot the test MSE values for each k.

Find the k value associated with the lowest test MSE.



# Import necessary libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
%matplotlib inline



#Read the file 'Advertising.csv' into a Pandas dataset
df = pd.read_csv('Advertising.csv')

#Take a quick look at the data
df.head()

#Set the 'TV' column as predictor variable
x = df['TV'].values.reshape(-1, 1)

#Set the 'Sales' column as response variable 
y = df['Sales'].values


###edTest(test_shape) ###
#Split the dataset in training and testing with 60% training set and 40% testing set 
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.6,random_state=66)

###edTest(test_nums) ###
#Choosing k range from 1 to 70
k_value_min = 1
k_value_max = 70

#Create a list of integer k values between k_value_min and k_value_max using linspace
k_list = np.linspace(k_value_min,k_value_max,num=70,dtype=int)

#Setup a grid for plotting the data and predictions
fig, ax = plt.subplots(figsize=(10,6))

#Create a dictionary to store the k value against MSE fit {k: MSE@k} 
knn_dict = {}

#Variable used for altering the linewidth of values kNN models
j=0

#Loop over all k values
for k_value in k_list:   
    
    # Create a KNN Regression model for the current k
    model = KNeighborsRegressor(n_neighbors=int(k_value))
    
    # Fit the model on the train data
    model.fit(x_train,y_train)
    
    # Use the trained model to predict on the test data
    y_pred = model.predict(x_test)
    
    # Calculate the MSE of the test data predictions
    MSE = mean_squared_error(y_test, y_pred)

    # Store the MSE values of each k value in the dictionary
    knn_dict[k_value] = MSE
    
    
    # Helper code to plot the data and various kNN model predictions
    colors = ['grey','r','b']
    if k_value in [1,10,70]:
        xvals = np.linspace(x.min(),x.max(),100).reshape(-1, 1)
        ypreds = model.predict(xvals)
        ax.plot(xvals, ypreds,'-',label = f'k = {int(k_value)}',linewidth=j+2,color = colors[j])
        j+=1
        
ax.legend(loc='lower right',fontsize=20)
ax.plot(x_train, y_train,'x',label='test',color='k')
ax.set_xlabel('TV budget in $1000',fontsize=20)
ax.set_ylabel('Sales in $1000',fontsize=20)
plt.tight_layout()

![image](https://github.com/user-attachments/assets/58f53f1b-33ba-4cb7-aa43-4ab6d1b073a5)


#Plot a graph which depicts the relation between the k values and MSE
plt.figure(figsize=(8,6))
plt.plot(list(knn_dict.keys()),list(knn_dict.values()),'k.-',alpha=0.5,linewidth=2)

#Set the title and axis labels
plt.xlabel('k',fontsize=20)
plt.ylabel('MSE',fontsize = 20)
plt.title('Test $MSE$ values for different k values - KNN regression',fontsize=20)
plt.tight_layout()



![image](https://github.com/user-attachments/assets/e17942b2-9c66-416e-aa77-9468284422be)

#### Find the best knn model
###edTest(test_mse) ###

#Find the lowest MSE among all the kNN models
min_mse = min(knn_dict.values())

#Use list comprehensions to find the k value associated with the lowest MSE
best_model = [key  for (key, value) in knn_dict.items() if value == min_mse]

#Print the best k-value
print ("The best k value is ",best_model,"with a MSE of ", min_mse)

##### The best k value is  [np.int64(9)] with a MSE of  13.046766975308643
From the options below, how would you classify the "goodness" of your model?
A. Good
B. Satisfactory
C. Bad
###edTest(test_chow1) ###
#Submit an answer choice as a string below (eg. if you choose option A, put 'A')
answer1 = 'B'

#Helper code to compute the R2_score of your best model
model = KNeighborsRegressor(n_neighbors=best_model[0])
model.fit(x_train,y_train)
y_pred_test = model.predict(x_test)

#Print the R2 score of the model
print(f"The R2 score for your model is {r2_score(y_test, y_pred_test)}")
##### The R2 score for your model is 0.5492457002030715

After observing the R^2  value, how would you now classify your model?
A. Good
B. Satisfactory
C. Bad
###edTest(test_chow2) ###
#Submit an answer choice as a string below (eg. if you choose option A, put 'A')
answer2 = 'B'
