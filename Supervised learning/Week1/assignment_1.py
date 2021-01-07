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

# Show pairwise relationships
sns.pairplot(advertising_data_frame)
plt.show()
