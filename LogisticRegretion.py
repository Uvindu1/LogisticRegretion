import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # meken thamai trenig and testing data kadaganne
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("sample.csv")
#print(data.head())    data set eke mula 5 genima
#print(data.tail())   data set eke aga 5 genima

plt.scatter(data.age,data.job)
plt.show()

x = data[["age"]]
y = data["job"]

x_train, x_test, y_train, y_test = train_test_split(x, y , test_size= 0.2)

model = LogisticRegression()

model.fit(x_train, y_train)

# cochchara durata sarthakada belima
# test karanna thibba x value tika dala y gaththa, ekath ekka data set eke y value tika sana sandanaya kara
Test = model.predict(x_test)
print(Test)

accurecy = model.score(x_test,y_test) # acckiyurasi eka belima
print(accurecy)

# aluth data deela belima
ages = np.array([[31],[22],[34]])
new_pre = model.predict(ages)
print(new_pre)