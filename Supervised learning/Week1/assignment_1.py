import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

advertising_data_frame = pd.read_csv('advertising.csv')

# Display first 5 rows
print(advertising_data_frame.head(), '\n')

# Display size of DataFrame
print(advertising_data_frame.shape, '\n')

# Display base statistics of DataFrame
print(advertising_data_frame.describe(), '\n')

# Show pairwise relationships as graphs between DataFrame features
# sns.pairplot(advertising_data_frame)
# plt.show()

# Show pairwise relationships as correlations between DataFrame features
print(advertising_data_frame.corr(), '\n')

print(
    'From graphs and correlation table can be seen, that TV most influences on Sales - 78.2224% (linear relationship)')

# Linear normalization to depression sigma: xi = (xi - mean(x)) / sigma
# Long way:
for column in advertising_data_frame:  # iterate each column
    mean = advertising_data_frame[column].mean()  # Calculate mean of column
    std = advertising_data_frame[column].std()  # Calculate std of column
    print(column + ' parameters: mean=', mean, ', std=', std)
    advertising_data_frame[column] = (advertising_data_frame[column] - mean) / std  # normalize each value in column

# short way:
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# print(pd.DataFrame(scaler.fit_transform(advertising_data_frame), columns=advertising_data_frame.columns).head())

print('\n')

# Display first 5 rows
print(advertising_data_frame.head(2), '\n')

# add column of ones as w0
advertising_data_frame['bias'] = 1
print(advertising_data_frame.head(2), '\n')


# realisation function 'mean_square_error': mean square error: sum((y[i] - y_predicted[i])^2) / n
# Long way:
def mean_square_error(y_true, y_predicted):
    """
    Calculate mean square error
    :param y_true: real result
    :param y_predicted: predicted result
    :return:
    """
    return ((y_true - y_predicted) ** 2).sum() / len(y_predicted)


# Short way:
# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(np.array([1, 1, 1]), np.array([1, 1, 1])))

# Test mean_square_error
print(mean_square_error(np.array([1, 1, 1]), np.array([1, 1, 1])))
print(mean_square_error(np.array([1, 1, 1]), np.array([1.1, 1.1, 1.1])))
print(mean_square_error(np.array([1, 1, 1]), np.array([2, 2, 2])), '\n')

# Calculate mean square error of Sales against median Sales
sales_median = advertising_data_frame.Sales.median()
print('Sales median: ', sales_median, ', MSE of median: ',
      mean_square_error(sales_median, advertising_data_frame.Sales), '\n')


# realization of normal_equation function, which calculates weights X * w = y
def normal_equation(x, y):
    a = np.dot(x.T, x)
    b = np.dot(x.T, y)
    return np.linalg.solve(a, b)


# testing normal_equation
# the weights very similar to correlation matrix!
feat_matrix = advertising_data_frame[['TV', 'Radio', 'Newspaper', 'bias']].values
target_matrix = advertising_data_frame.Sales
norm_eq_weights = normal_equation(feat_matrix, target_matrix)
print(norm_eq_weights, '\n')


# realization of function wich calculate prediction from features and weights
def linear_prediction(X, w):
    return np.dot(X, w)


# lets see prediction and calculate error
y_pred = linear_prediction(feat_matrix, norm_eq_weights)
print('MSE of norm.eq: ', mean_square_error(target_matrix, y_pred), '\n')

# mean almost zero:
print(advertising_data_frame.TV.mean())
print(advertising_data_frame.Radio.mean())
print(advertising_data_frame.Newspaper.mean(), '\n')

# calculate MSE in case of mean TV, Radio and Newspaper aiming to 0
mean_values = np.array([0, 0, 0, 1]).T
y_pred = linear_prediction(mean_values, norm_eq_weights)
print(y_pred, '\n')
