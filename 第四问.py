import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.colors import LinearSegmentedColormap
import math
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import matplotlib.cm as cmx
import matplotlib
from sklearn.svm import SVR

#4-2
#最优操作选取
x = pd.read_excel('x.xlsx').values
y = pd.read_excel('y.xlsx').values
y=y.ravel()
x_train = x[:-81]
y_train = y[:-81]
x_test = x[-81:]
y_test = y[-81:]
clf = SVR(kernel='linear')
clf.fit(x_train, y_train)
df = pd.read_excel('xr.xlsx')
columns = list(df.columns)
error = 0
x133 = df.iloc[132, :].values
y = pd.read_excel('y.xlsx').values
for a in np.arange(2.35, 2.7, 0.1):
    for b in np.arange(0, 121, 10):
        for c in np.arange(2, 101, 10):
            for d in np.arange(3, 76, 4):
                for e in np.arange(320, 482, 20):
                    for f in np.arange(0.4, 0.9, 0.1):
                        df[columns[-6]] = [(a-2.3859664)/(2.607782-2.3859664) for i in range(df.shape[0])]
                        df[columns[-5]] = [(b-0.3016754)/(100.81921-0.3016754) for i in range(df.shape[0])]
                        df[columns[-4]] = [(c-2.8080836)/(83.222635-2.8080836) for i in range(df.shape[0])]
                        df[columns[-3]] = [(d-3.6843995)/(64.396493-3.6843995) for i in range(df.shape[0])]
                        df[columns[-2]] = [(e-334.99402)/(457.82386-334.99402) for i in range(df.shape[0])]
                        df[columns[-1]] = [(f-0.4305181)/(0.6833051-0.4305181) for i in range(df.shape[0])]
                        x1 = df.values
                        # print(x1.shape)
                        y_pred = clf.predict(x1)
                        # plt.plot(y_pred, color='red', label='Predict RONloss')
                        # plt.plot(y, color='blue', label='Real RONloss')
                        # plt.title('RONloss Prediction')
                        # plt.show()
                        if error < (np.mean((y-y_pred)/y)):
                            params = (a, b, c, d, e, f)
                            error = (np.mean((y-y_pred)/y))
                            print('******', params, (np.mean((y-y_pred)/y)))
                            print('******', y[132], '---->', clf.predict(x133[np.newaxis, :]))
            print((a, b, c, d, e, f))
            print('=========')
print('*******')
print(params)

df = pd.read_excel('xr.xlsx')
columns = list(df.columns)
y = pd.read_excel('y.xlsx').values
a, b, c, d, e, f = params[0], params[1], params[2], params[3], params[4], params[5]
df[columns[-6]] = [(a-2.3859664)/(2.607782-2.3859664) for i in range(df.shape[0])]
df[columns[-5]] = [(b-0.3016754)/(100.81921-0.3016754) for i in range(df.shape[0])]
df[columns[-4]] = [(c-2.8080836)/(83.222635-2.8080836) for i in range(df.shape[0])]
df[columns[-3]] = [(d-3.6843995)/(64.396493-3.6843995) for i in range(df.shape[0])]
df[columns[-2]] = [(e-334.99402)/(457.82386-334.99402) for i in range(df.shape[0])]
df[columns[-1]] = [(f-0.4305181)/(0.6833051-0.4305181) for i in range(df.shape[0])]
x = df.values
y_pred = clf.predict(x)
print(np.mean((y-y_pred)/y))
print(y_pred[132])
plt.plot(y_pred, color='red', label='Predict')
plt.plot(y, color='blue', label='Real')
plt.legend(['Predict', 'Real'])
plt.title('RONloss Prediction')
plt.show()



