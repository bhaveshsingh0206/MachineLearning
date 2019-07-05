import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

temp1 = pd.read_excel('Final_train.xlsx')
temp2 = pd.read_excel('Final_test.xlsx')

def calculateDegress(dataframe):
    degree = dataframe.Qualification.apply(lambda x: len(x.split(',')))
    return degree

train_degree = calculateDegress(temp1)
test_degree = calculateDegress(temp2)
temp1["Number_of_degree"] =  train_degree
temp2["Number_of_degree"] =  test_degree

experience_train = temp1.Experience.apply(lambda x: x[0:(x.index(' '))])
temp1["Experience"] =  experience_train
temp1.Experience=pd.to_numeric(temp1["Experience"])

experience_test = temp2.Experience.apply(lambda x: x[0:(x.index(' '))])
temp2["Experience"] =  experience_test
temp2.Experience=pd.to_numeric(temp2["Experience"])

#Missing value
temp1['Rating'].fillna('0%',inplace = True)
temp1['Rating'] = temp1['Rating'].str.slice(stop=-1).astype(int)
temp2['Rating'].fillna('0%',inplace = True)
temp2['Rating'] = temp2['Rating'].str.slice(stop=-1).astype(int)

X_train = temp1.iloc[:,[1,2,4,7]].values
X_test = temp2.iloc[:,[1,2,4,6]].values
y_train =temp1.iloc[:,6].values


#Label Encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()

X_train[:,2] = labelencoder.fit_transform(X_train[:,2])
X_test[:,2] = labelencoder.transform(X_test[:,2])
onehotencoder = OneHotEncoder(categorical_features = [2])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.transform(X_test).toarray()


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)






import statsmodels.formula.api as sm
def backwardElimination(x, SL,y):
    numVars = len(x[0])
    temp = np.zeros((x.shape[0],9)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X_train[:,[0,1,2,3,4,5,6,7,8]]
X_Modeled = backwardElimination(X_opt, SL,y_train)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Modeled,y_train)
y_pred = regressor.predict(X_test)