# Section 2: Multiple and Polynomial Regression
## Ex 1: 
The aim of this exercise is to understand how to use Multi-Linear Regression. Here we will compare simple Linear Regression models consisting of different columns with a 
Multi-linear Regression model comprising of all columns.
### Instructions:
Read the file Advertisement.csv as a Pandas dataframe.Identify the predictor and response variables from the dataframe.Initialize an empty Pandas dataframe df_results to
store the R^2 values of each model fitted.

Call the helper function fit_and_plot_linear with one predictor/feature (as a parameter) at a time.

The function will split the data into train and test sets (which will be covered in depth in future sessions).

Train a linear model on the train data.

Predict on both the train and test data.

Plot the train and test data along with the model predictions.

Compute and return the R^2 on the train and test set.


Call the helper function fit_and_plot_multi with no parameters.

The function will split the data into train and test sets (which will be covered in depth in future sessions)

Train a linear model on the train data with all the predictors.

Predict on both the train and test data.

Plot the train and test data along with the model predictions.

Compute and return the R^2 on the train and test set.

Update the dataframe df_results with the R^2 from all 4 models.
__________________________________________________________________________________________________________________________________
pd.read_csv(filename)
 Returns a pandas dataframe containing the data and labels from the file data
FUNCTION SIGNATURE

fit_and_plot_linear(df[['predictor']]):Returns the R^2 compute on the train data followed by the R^2  compute on the test data. 


fit_and_plot_multi():Returns the R^2 compute on the train data followed by the R^2 compute on the test data. 
___________________________________________________________________________________________________________________________________

#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import fit_and_plot_linear, fit_and_plot_multi
%matplotlib inline
#Read the file "Advertising.csv"
df = pd.read_csv("Advertising.csv")

#Take a quick look at the dataframe
df.head()

#Define an empty Pandas dataframe to store the R-squared value associated with each predictor for both the train and test split
df_results = pd.DataFrame(columns=['Predictor', 'R2 Train', 'R2 Test'])

#For each predictor in the dataframe, call the function "fit_and_plot_linear()" from the helper file with the predictor as a parameter to the function
This function will split the data into train and test split, fit a linear model on the train data and compute the R-squared value on both the train and test data

**Your code here**
#Define the predictors from the dataframe
predictors = ['TV', 'Radio', 'Newspaper']

#Loop through each predictor
for predictor in predictors:
    # Call the helper function for each predictor and get the R^2 values
    r2_train, r2_test = fit_and_plot_linear(predictor)
    
    # Append the results to the dataframe
    df_results = df_results.append({
        'Predictor': predictor,
        'R2 Train': r2_train,
        'R2 Test': r2_test
    }, ignore_index=True)

# Unfinished


### EX:2 Regression Model
The aim of this exercise is to understand how to use multi regression. Here we will observe the difference in MSE for each model as the predictors change. 
Instructions:
Read the file Advertisement.csv as a dataframe.

For each instance of the predictor combination, we will form a model. For example, if you have 2 predictors,  A and B, you will end up getting 3 models - one with only A, one with only B, and one with both A and B.

Split the data into train and test sets.

fit a linear regression model on the train data.

Compute the MSE of each model on the test data.

Print the results for each Predictor - MSE value pair.
__________________________________________________________________________________________________________
pd.read_csv(filename): Returns a pandas dataframe containing the data and labels from the file data.


sklearn.model_selection.train_test_split():Splits the data into random train and test subsets.


sklearn.linear_model.LinearRegression():LinearRegression fits a linear model.


sklearn.linear_model.LinearRegression.fit():Fits the linear model to the training data.


sklearn.linear_model.LinearRegression.predict():Predict using the linear model.


sklearn.metrics.mean_squared_error():Computes the mean squared error regression loss
__________________________________________________________________________________________________________
#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
%matplotlib inline

#Read the file "Advertising.csv"
df = pd.read_csv("Advertising.csv")
#Take a quick look at the data to list all the predictors
df.head()

#### Create different multi predictor models
###edTest(test_mse) ###

#Initialize a list to store the MSE values
mse_list = []

#List of all predictor combinations to fit the curve
cols = [['TV'],['Radio'],['Newspaper'],['TV','Radio'],['TV','Newspaper'],['Radio','Newspaper'],['TV','Radio','Newspaper']]

#Loop over all the predictor combinations 
for i in cols:

    # Set each of the predictors from the previous list as x
    x = df[i]
    
    
    # Set the "Sales" column as the reponse variable
    y = df["Sales"]
    
   
    # Split the data into train-test sets with 80% training data and 20% testing data. 
    # Set random_state as 0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Initialize a Linear Regression model
    lreg = LinearRegression()

    # Fit the linear model on the train data
    lreg.fit(x_train, y_train)
    
    # Predict the response variable for the test set using the trained model
    y_pred= lreg.predict(x_test)
    
    # Compute the MSE for the test data
    MSE = mean_squared_error(y_test, y_pred)
    
    # Append the computed MSE to the list
    mse_list.append(MSE)

#Helper code to display the MSE for each predictor combination
t = PrettyTable(['Predictors', 'MSE'])

for i in range(len(mse_list)):
    t.add_row([cols[i],mse_list[i]])

print(t)
+------------------------------+--------------------+
|          Predictors          |        MSE         |
+------------------------------+--------------------+
|            ['TV']            | 10.186181934530222 |
|          ['Radio']           | 24.23723303713214  |
|        ['Newspaper']         | 32.13714634300907  |
|       ['TV', 'Radio']        | 4.3914297635818835 |
|     ['TV', 'Newspaper']      | 8.687682675690592  |
|    ['Radio', 'Newspaper']    | 24.783395482938165 |
| ['TV', 'Radio', 'Newspaper'] | 4.402118291449691  |
+------------------------------+--------------------+



#### Ex: 3 A Line Won't Cut It
Here we encounter a non-linear dataset that motivates model of higher degree.
We'll judge that a the linear model is ill-suited for this data by plotting the model's predictions and inspecting its residuals. 
Instructions:
Read the poly.csv file into a dataframe.
Split the data into train and test subsets.
Fit a linear regression model on the entire data, using LinearRegression() object from Sklearn library.
Guesstimate the degree of the polynomial which would best fit the data.
Fit a polynomial regression model on the computed Polynomial Features using LinearRegression() object from sklearn library.
Plot the linear and polynomial model predictions along with the test data.
Compute the polynomial and linear model residuals using the formula below 
![image](https://github.com/user-attachments/assets/696476e7-6943-4ba4-bcb3-b5c3451935a9)

Plot the histogram of the residuals and comment on your choice of the polynomial degree.
______________________________________________________________________________________________________________________________________
pd.DataFrame.head() Returns a pandas dataframe containing the data and labels from the file data.

sklearn.model_selection.train_test_split() Splits the data into random train and test subsets.

plt.subplots() Create a figure and a set of subplots.

sklearn.preprocessing.PolynomialFeatures() Generates a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree.

sklearn.preprocessing.StandardScaler.fit_transform() Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X.

sklearn.linear_model.LinearRegression LinearRegression fits a linear model.

sklearn.linear_model.LinearRegression.fit() Fits the linear model to the training data.

sklearn.linear_model.LinearRegression.predict() Predict using the linear model.

plt.plot() Plots x versus y as lines and/or markers.

plt.axvline() Add a vertical line across the axes.

ax.hist() Plots a histogram.

______________________________________________________________________________________________________________________________________
#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
%matplotlib inline


#Read the data from 'poly.csv' into a Pandas dataframe
df = pd.read_csv('poly.csv')

#Take a quick look at the dataframe
df.head()
#Get the column values for x & y as numpy arrays
x = df[['x']].values
y = df['y'].values

#Helper code to plot x & y to visually inspect the data
fig, ax = plt.subplots()
ax.plot(x,y,'x')
ax.set_xlabel('$x$ values')
ax.set_ylabel('$y$ values')
ax.set_title('$y$ vs $x$')
plt.show();

![image](https://github.com/user-attachments/assets/8baf175e-f59f-4f99-9404-4667588d46e2)

#Split the data into train and test sets
#Set the train size to 0.8 and random state to 22
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=22)


#Initialize a linear model
model = LinearRegression()

#Fit the model on the train data
model.fit(x_train, y_train)

#Get the predictions on the test data using the trained model
y_lin_pred = model.predict(x_test)

#Helper code to plot x & y to visually inspect the data
fig, ax = plt.subplots()
ax.plot(x,y,'x', label='data')
ax.set_xlabel('$x$ values')
ax.set_ylabel('$y$ values')
ax.plot(x_test, y_lin_pred, label='linear model predictions')
plt.legend();
![image](https://github.com/user-attachments/assets/5ee3d7cf-4940-453e-a466-1f9042885b92)

Above is the line that minimizes MSE of the training data. How would you describe the results?

A. The model would perform better with more training data
B. This line is a good fit to the data
C. No straight line will fit this data well
###edTest(test_chow1) ###
#Type your answer within in the quotes given
answer1 = 'C'
###edTest(test_deg) ###

###edTest(test_deg) ###
#Guess the correct polynomial degree based on the above graph
guess_degree = 3

#Predict on the entire polynomial transformed test data using helper function.
y_poly_pred = get_poly_pred(x_train, x_test, y_train, degree=guess_degree) 
#Generate polynomial features on the test data
x_poly_test= PolynomialFeatures(degree=guess_degree).fit_transform(x_test)


#Helper code to visualise the results
idx = np.argsort(x_test[:,0])
x_test = x_test[idx]

#Use the above index to get the appropriate predicted values for y
#y values corresponding to sorted test data
y_test = y_test[idx]

#Linear predicted values  
y_lin_pred = y_lin_pred[idx]

#Non-linear predicted values
y_poly_pred= y_poly_pred[idx]


#First plot x & y values using plt.scatter
plt.scatter(x, y, s=10, label="Test Data")

#Plot the linear regression fit curve
plt.plot(x_test,y_lin_pred,label="Linear fit", color='k')

#Plot the polynomial regression fit curve
plt.plot(x_test, y_poly_pred, label="Polynomial fit",color='red', alpha=0.6)

#Assigning labels to the axes
plt.xlabel("x values")
plt.ylabel("y values")
plt.legend()
plt.show()

###edTest(test_poly_predictions) ###
#Calculate the residual values for the polynomial model
poly_residuals = y_test - y_poly_pred


###edTest(test_linear_predictions) ###
#Calculate the residual values for the linear model
lin_residuals = y_test - y_lin_pred


# Helper code to plot the residual values
# Plot the histograms of the residuals for the two cases

#Distribution of residuals
fig, ax = plt.subplots(1,2, figsize = (10,4))
bins = np.linspace(-20,20,20)
ax[0].set_xlabel('Residuals')
ax[0].set_ylabel('Frequency')

#Plot the histograms for the polynomial regression
ax[0].hist(poly_residuals, bins, label = 'poly residuals', color='#B2D7D0', alpha=0.6)

#Plot the histograms for the linear regression
ax[0].hist(lin_residuals, bins, label = 'linear residuals', color='#EFAEA4', alpha=0.6)

ax[0].legend(loc = 'upper left')

#Distribution of predicted values with the residuals
ax[1].hlines(0,-75,75, color='k', ls='--', alpha=0.3, label='Zero residual')
ax[1].scatter(y_poly_pred, poly_residuals, s=10, color='#B2D7D0', label='Polynomial predictions')
ax[1].scatter(y_lin_pred, lin_residuals, s = 10, color='#EFAEA4', label='Linear predictions' )
ax[1].set_xlim(-75,75)
ax[1].set_xlabel('Predicted values')
ax[1].set_ylabel('Residuals')
ax[1].legend(loc = 'upper left')
fig.suptitle('Residual Analysis (Linear vs Polynomial)')
plt.show();


![image](https://github.com/user-attachments/assets/07c77c08-dc2a-4a0a-8032-6ed94ab48707)

What is it about the plots above that are sign that a linear model is not appropriate for the data?

A. Residuals not normally distributed
B. Residuals distribution not clearly centered on zero
C. Residuals do not have constant variance
D. All of the above

###edTest(test_chow2) ###
#Type your answer within in the quotes given
answer2 = 'D'
__________________________________________________________________________________________________________
Exercise 3: Miltiple regression 
Polynomial Modeling
The goal of this exercise is to fit linear regression and polynomial regression to the given data. Plot the fit curves of both the models along with the data and observe what the residuals tell us about the two fits. 
Instructions
Read the poly.csv file into a dataframe

Fit a linear regression model on the entire data, using LinearRegression() object from sklearn library

Guesstimate the degree of the polynomial which would best fit the data

Fit a polynomial regression model on the computed PolynomialFeatures using LinearRegression() object from sklearn library

Plot the linear and polynomial model predictions along with the data

Compute the polynomial and linear model residuals using the formula below 
![image](https://github.com/user-attachments/assets/df69b4eb-2ea6-4d0d-8fbb-b3fdc5ce5f6d)

Plot the histogram of the residuals and comment on your choice of the polynomial degree. 

Hints:
df.head()
 Returns a pandas dataframe containing the data and labels from the file data

plt.subplots()
Create a figure and a set of subplots

sklearn.PolynomialFeatures()
Generates a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree

sklearn.fit_transform()
Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X

sklearn.LinearRegression()
LinearRegression fits a linear model

sklearn.fit()
Fits the linear model to the training data

sklearn.predict()
Predict using the linear model

plt.plot()
Plots x versus y as lines and/or markers

plt.axvline()
Add a vertical line across the axes

ax.hist()
Plots a histogram
________________________________________________________________________________________________________________________________________
SOlution 
#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
%matplotlib inline

#Read the data from 'poly.csv' to a dataframe
df = pd.read_csv('poly.csv')
# Get the column values for x & y in numpy arrays
x = df[['x']].values
y = df['y'].values

#Take a quick look at the dataframe
df.head()
#Plot x & y to visually inspect the data

fig, ax = plt.subplots()
ax.plot(x,y,'x')
ax.set_xlabel('$x$ values')
ax.set_ylabel('$y$ values')
ax.set_title('$y$ vs $x$');

#Set the train size to 0.8 and random state to 22
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=22)
#Initialize a linear model
model = LinearRegression()

#Fit the model on the train data
model.fit(x_train, y_train)

#Get the predictions on the test data using the trained model
y_lin_pred = model.predict(x_test)

#Guess the correct polynomial degree based on the above graph
guess_degree = 4

#Generate polynomial features on the train data
x_poly_train= PolynomialFeatures(degree=guess_degree).fit_transform(x_train)

#Generate polynomial features on the test data
x_poly_test= PolynomialFeatures(degree=guess_degree).fit_transform(x_test)

#Initialize a model to perform polynomial regression
polymodel = LinearRegression(fit_intercept=False)




