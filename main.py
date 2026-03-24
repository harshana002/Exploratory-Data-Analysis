import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load Titanic dataset
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df= pd.read_csv(url)

#Inspect Data
print(df.info())
print(df.describe())

#Handle missiong values
df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

#Remove duplicates
df= df.drop_duplicates()

#Filter data: Passengers in first class
first_class= df[df["Pclass"]==1]
print("First Class Passengers: \n", first_class.head())

#Bar Chart: Servival rate by Chart
survival_by_Class= df.groupby("Pclass")["Survived"].mean()
survival_by_Class.plot(kind="bar", color="skyblue")
plt.title("Survival Rate by class")
plt.ylabel("Survival Rate")
plt.show()

#Histogram: Age Distribution
sns.histplot(df["Age"],kde=True,bins=20,color="purple")
plt.title("Age DIstribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

#Scatter Plot: AGe vs Fare
plt.scatter(df["Age"],df["Fare"],alpha=0.5, color="green")
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()