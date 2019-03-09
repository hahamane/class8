#from sklearn.datasets import load_iris
#type(load_iris)
#iris_data=load_iris()
#dir(iris_data)
#print(iris_data['DESCR']
#data=iris_data['data']
#type(data)
#[func for func in dir(sklearn.datasets) if func.startswith("load")]
#'DESCR' : This is a description of the dataset
#'data'  : This is the numpy dataset
#'data.filename' : This is the location of the package
#'data.feature_names' : This is the header of the columns
#'target' : This is the value of the thing that we are trying to predict

from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import neighbors, datasets
data = load_diabetes()
#dataPD=pd.DataFrame(data=data["data"], columns=data["feature_names"])
#print(dataPD)

x_index = 4
y_index = 5
#formatter = plt.FuncFormatter(*args:[1,2])
#plt.figure(figsize=(5,4))
#plt.scatter(data.data[:,x_index],data.data[:,y_index], c=data.data[:,1],cmap='viridis')
#clb=plt.colorbar()
#clb.ax.set_title("SEX")
#plt.xlabel(data.feature_names[x_index])
#plt.ylabel(data.feature_names[y_index])

#plt.tight_layout()
#plt.show()

model = LinearRegression(normalize =True)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data.data, data.target)
clf = LinearRegression()
clf.fit(x_train,y_train)
predicted = clf.predict(x_test)
expected = y_test
plt.scatter(expected,predicted)
plt.show()
