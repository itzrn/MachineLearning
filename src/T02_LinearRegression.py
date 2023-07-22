import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing Dataset
dataset = pd.read_csv("Datasets/T02_Salary_Data.csv")
x_feature = dataset.iloc[:, :-1].values
y_outcome = dataset.iloc[:, [-1]].values

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(x_feature, y_outcome, test_size=0.2, random_state=1)
# print(X_train)
# print()
# print(y_train)

# Training the simple Linear regression model on the training set
# regression is done on continuous data
linear_regressor = LinearRegression()
"""
This will just create the simple linear regression model,
and it is so simple that usually, we don't have to
play too much with the parameters, alright?
And that line of code
directly creates the simple linear regression model.
And that's only the building part.
You know, we actually get a model.
But now, of course, we have to train it
on the training set and therefore we have to connect it
in some way to the training set.
And the action, or you know the function that connects it
in order to train it is called the fit function.
"""
linear_regressor.fit(X_train, y_train)
# this fit method will train the simple linear regression model on the giving training dataset


# predicting the result using test set and evaluating the model
y_pred = linear_regressor.predict(X_test)  # y_pred contain the predicted Salary

# Visualising the TRAINING SET result
plt.scatter(X_train, y_train, color='red')  # this method will plot point the plane
plt.plot(X_train, linear_regressor.predict(X_train), color='blue')  # this function will plot the line
plt.title('Salary Vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the TEST SET result
plt.scatter(X_test, y_test, color='red')
# plt.plot(X_test, y_pred, color='blue')
# here the regression line will not get change even if we run the above line
# so, down line cal also we done
plt.plot(X_train, linear_regressor.predict(X_train), color='blue')
plt.title('Salary Vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# to get the coefficient of the linear regression line
# here coefficient is slope -> What dose it tell?
# it tells that how much the dependent variable depends on independent variable
print("Coefficient ->\n", linear_regressor.coef_)

# to get the intercept of the linear regression line
print("Intercept ->\n", linear_regressor.intercept_)

print("Predicted The Salary Of Employ Having 12 Year Experience ->\n", linear_regressor.predict([[12]]))
