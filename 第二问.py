#2-3
#LASSOLARS
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt  # 可视化绘制
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV,ElasticNet, ElasticNetCV

columns = list(pd.read_excel('data_normal.xlsx').columns)
x_train = pd.read_excel('操作变量.xlsx').values[:276, :]
y_train = pd.read_excel('y.xlsx').values[:276]
x_test = pd.read_excel('操作变量.xlsx').values[276:, :]
y_test = pd.read_excel('y.xlsx').values[276:]

def lassolarscv():
	global x_train, y_train, x_test, y_test
	lasso = LassoLarsCV().fit(x_train, y_train)
	print('---------')
	print(lasso.alpha_)
	print(lasso.coef_)
	return lasso
lassolarscv()

#2-4
#ElasticNet
def elasticnetcv():
	global x_train, y_train, x_test, y_test
	lasso = ElasticNetCV().fit(x_train, y_train)
	print('---------')
	print(lasso.alpha_)
	#print(lasso.coef_)

	elastic = elasticnetcv()
	coef = elastic.coef_
	inte = elastic.intercept_
	print ("training set score:{:.2f}".format(elastic.score(x_train,y_train)))
	print ("test set score:{:.2f}".format(elastic.score(x_test,y_test)))
	keyp =[]
	for i in range(len(coef)):
		if coef[i] != 0:
			keyp.append(columns[i])
	print(keyp)
	return lasso
elasticnetcv()

#2-5
#RandomForestClassifier
def randomforest():
	clf = RandomForestClassifier()
	df = pd.read_excel('样本数据z_normal.xlsx')
	x = df.values
	y = pd.read_excel('y.xlsx').values
	clf.fit(x, y.astype('int'))
	importance = clf.feature_importances_
	indices = np.argsort(importance)[::-1]
	features = df.columns
	for f in range(x.shape[1]):
		print(("%-*s %f" % (30, features[f], importance[indices[f]])))
randomforest()










