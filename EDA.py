import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = sns.load_dataset('iris')

# Display the first few rows of the dataset
print(df.head())

# Understand the dataset
# Display basic information about the dataset
print(df.info())

# Display summary statistics
print(df.describe())

# Clean the data
print(df.isnull().sum())

# Explore basic statistics
# Summary statistics
print(df.describe())

# Count of each species
print(df['species'].value_counts())

# Visualize distributions
# Histograms
df.hist(bins=20, figsize=(10, 10))
plt.show()

# Boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.show()

# Pairplot to visualize pairwise relationships
sns.pairplot(df, hue='species')
plt.show()

# Analyze relationships and correlations
# Scatter plot for Sepal Length vs Sepal Width
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df)
plt.show()

# Scatter plot for Petal Length vs Petal Width
sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=df)
plt.show()

# Drop non-numeric columns for correlation matrix
numeric_df = df.drop(columns=['species'])

# Correlation heatmap
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Identify outliers using box plots
# Boxplot for Sepal Length
sns.boxplot(x='species', y='sepal_length', data=df)
plt.show()

# Boxplot for Sepal Width
sns.boxplot(x='species', y='sepal_width', data=df)
plt.show()

# Boxplot for Petal Length
sns.boxplot(x='species', y='petal_length', data=df)
plt.show()

# Boxplot for Petal Width
sns.boxplot(x='species', y='petal_width', data=df)
plt.show()
