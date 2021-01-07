import numpy as np
import pandas as pd

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
# scaler = StandardScaler()
# print(pd.DataFrame(scaler.fit_transform(advertising_data_frame), columns=advertising_data_frame.columns).head())

print('\n')

# Display first 5 rows
print(advertising_data_frame.head(2), '\n')

# add column of ones as w0
advertising_data_frame['bias'] = 1
print(advertising_data_frame.head(2), '\n')


# realisation function 'mean_square_error': mean square error: sum((y[i] - y_predicted[i])^2) / n
def mean_square_error(y_true, y_predicted):
    """
    Calculate mean square error
    :param y_true: real result
    :param y_predicted: predicted result
    :return:
    """
    return ((y_true - y_predicted) ** 2).sum() / len(y_true)


# Test mean_square_error
print(mean_square_error(np.array([1, 1, 1]), np.array([1, 1, 1])))
print(mean_square_error(np.array([1, 1, 1]), np.array([1.1, 1.1, 1.1])))
print(mean_square_error(np.array([1, 1, 1]), np.array([2, 2, 2])))
