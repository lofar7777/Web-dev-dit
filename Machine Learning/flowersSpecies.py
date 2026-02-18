import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

data = pd.DataFrame(iris.data, columns = iris.feature_names)
print("Iris Dataset(First 5 rows): ")
print(data.head())

print("\nMean:")
print(data.mean())

print("\nVariance:")
print(data.var())

print("\nStandard Deviation:")
print(data.std())

print("\nDataset Summary (Min, 25%, 50%, 75%, Max):")
print(data.describe())


------------------------------------------

Iris Dataset(First 5 rows): 
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                1.5               0.2
4                5.0               3.6                1.4               0.2

Mean:
sepal length (cm)    5.843333
sepal width (cm)     3.057333
petal length (cm)    3.758000
petal width (cm)     1.199333
dtype: float64

Variance:
sepal length (cm)    0.685694
sepal width (cm)     0.189979
petal length (cm)    3.116278
petal width (cm)     0.581006
dtype: float64

Standard Deviation:
sepal length (cm)    0.828066
sepal width (cm)     0.435866
petal length (cm)    1.765298
petal width (cm)     0.762238
dtype: float64

Dataset Summary (Min, 25%, 50%, 75%, Max):
       sepal length (cm)  sepal width (cm)  petal length (cm)  \
count         150.000000        150.000000         150.000000   
mean            5.843333          3.057333           3.758000   
std             0.828066          0.435866           1.765298   
min             4.300000          2.000000           1.000000   
25%             5.100000          2.800000           1.600000   
50%             5.800000          3.000000           4.350000   
75%             6.400000          3.300000           5.100000   
max             7.900000          4.400000           6.900000   

       petal width (cm)  
count        150.000000  
mean           1.199333  
std            0.762238  
min            0.100000  
25%            0.300000  
50%            1.300000  
75%            1.800000  
max            2.500000 


---------------------------------------------

2nd-> 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#Load dataset
iris = load_iris()


#DataFrame
data = pd.DataFrame(iris.data, columns = iris.feature_names)

#Add Spacies column
data["species"] = iris.target

print("Species Count: ")
print(data["species"].value_counts())

#1. Boxplot
data.boxplot()
plt.title("Boxplot of Iris Features")
plt.show()

#2. Histogram
data.hist()
plt.show()

#3. Bar Plot (Species Count)
data["species"].value_counts().plot(kind="bar")
plt.title("Bar plot of Species")
plt.show()

#4. Scatter plot
plt.scatter(data["sepal length (cm)"], data["petal length (cm)"])
plt.title("Scatter plot")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()

# 5. Two-Way Table
table = pd.crosstab(data["species"], data["sepal width (cm)"] > 3)
print("\nTwo-way Table (Species vs Sepal Width > 3): ")
print(table)







------------

Species Count: 
2    50
1    50
0    50
Name: species, dtype: int64





Two-way Table (Species vs Sepal Width > 3): 
sepal width (cm)  False  True 
species                       
0                     8     42
1                    42      8
2                    33     17
