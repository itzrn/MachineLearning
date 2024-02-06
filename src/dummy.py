import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Creating sample data with three features
np.random.seed(42)
data = pd.DataFrame(np.random.randn(100, 3), columns=['Feature1', 'Feature2', 'Feature3'])

# Pair plot to visualize relationships between features
sns.pairplot(data)
plt.show()

# Correlation matrix heatmap
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
