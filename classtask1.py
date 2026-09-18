import numpy as np
import pandas as pd



import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv")
 
df.head(10)

df.sample(10)
df.info()
print(df.shape)
print(df.notnull().sum())
print("Duplicate Rows:", df.duplicated().sum())
print(df["Survived"].unique())
print(df.nunique())
print(df.describe())
print(df.isnull().sum())
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0].sort_values(ascending=False))
missing_percentage = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": missing_percentage
})
print(missing_df)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as lb_sns  # ya sirf import seaborn as sns

# Task 16: Remove Duplicate Rows
df = df.drop_duplicates().copy()

# Task 17: Handle Missing Age Values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Task 18: Handle Missing Embarked Values
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Task 19: Verify Missing Values After Cleaning
print("Missing values after cleaning:")
print(df.isnull().sum())

# Task 20: Calculate the Overall Survival Rate
survival_rate = df["Survived"].mean() * 100
print(f"Overall Survival Rate: {survival_rate:.2f}%")

# Task 21: Compare Survival Rate by Sex
print("\nSurvival Rate by Sex:")
print(df.groupby("Sex")["Survived"].mean() * 100)

# Task 22: Compare Survival Rate by Passenger Class
print("\nSurvival Rate by Passenger Class:")
print(df.groupby("Pclass")["Survived"].mean() * 100)

# Task 23: Create Histograms for Numerical Features
numeric_cols = ["Age", "Fare", "SibSp", "Parch"]
df[numeric_cols].hist(bins=30, figsize=(10, 6), layout=(2, 2))
plt.tight_layout()
plt.show()

# Task 24: Detect Possible Outliers Using Boxplots
plt.figure(figsize=(8, 4))
lb_sns.boxplot(data=df[numeric_cols])
plt.title("Boxplots of Numerical Features")
plt.show()

# Task 25: Create a Correlation Matrix
selected_cols = ["Age", "SibSp", "Parch", "Pclass", "Fare"]
print("\nCorrelation Matrix:")
print(df[selected_cols].corr())

# Task 26: Display Correlation as a Heatmap
plt.figure(figsize=(8, 6))
lb_sns.heatmap(
    df[["Age", "SibSp", "Parch", "Pclass", "Fare"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
)
plt.title("Correlation Heatmap")
plt.show()

# Task 27: Compare Survival by Sex Using a Bar Plot
lb_sns.barplot(x="Sex", y="Survived", data=df, hue="Sex", legend=False)
plt.title("Sex vs Survival")
plt.xlabel("Sex")
plt.ylabel("Average Survival Rate")
plt.show()

# Task 28: Find the Five Highest-Paying Passengers
topFares = df.sort_values(by="Fare", ascending=False)
top5 = topFares.head(5)
print("\nTop 5 Highest-Paying Passengers:")
print(top5[["Name", "Sex", "Pclass", "Fare"]])

# Task 29: Visualize the Five Highest-Paying Passengers
lb_sns.barplot(x="Fare", y="Name", data=top5, hue="Sex")
plt.title("Five Highest-Paying Passengers")
plt.xlabel("Fare")
plt.ylabel("Passenger Name")
plt.show()