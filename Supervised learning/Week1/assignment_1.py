import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

advertising_data_frame = pd.read_csv('advertising.csv')

# Display first 5 rows
print(advertising_data_frame.head(), '\n')

# Display size of DataFrame
print(advertising_data_frame.shape, '\n')

# Display base statistics of DataFrame
print(advertising_data_frame.describe(), '\n')

# Show pairwise relationships as graphs between DataFrame features
sns.pairplot(advertising_data_frame)
plt.show()

# Show pairwise relationships as correlations between DataFrame features
print(advertising_data_frame.corr(), '\n')

print('From graphs and correlation table can be seen, that TV most influences on Sales')
