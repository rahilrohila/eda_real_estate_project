
# eda_real_estate_project.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Data
df = pd.read_csv('housing_data.csv')
print("Initial Data Loaded")

# 2. Clean the Data
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# 3. Univariate Analysis
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], kde=True)
plt.title('House Price Distribution')
plt.savefig('price_distribution.png')

# 4. Multivariate Analysis
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')

# 5. Feature Engineering
df['price_per_sqft'] = df['price'] / df['total_sqft']
df['property_age'] = 2025 - df['year_built']

# 6. Size Impact Analysis
sns.scatterplot(data=df, x='total_sqft', y='price')
plt.title('Size vs Price')
plt.savefig('size_vs_price.png')

df.to_csv('final_cleaned_data.csv', index=False)
