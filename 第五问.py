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

x = pd.read_excel('x.xlsx').values
y = pd.read_excel('y.xlsx').values
y=y.ravel()
x_train = x[:-81]
y_train = y[:-81]
x_test = x[-81:]
y_test = y[-81:]
clf = SVR(kernel='linear')
clf.fit(x_train, y_train) 
params = (2.65, 0, 92, 3, 480, 0.8)
delta = (0.1, 10, 1, 1, 1, 2, 0.1)
df = pd.read_excel('x133.xlsx')
x133 = pd.read_excel('x133_normal.xlsx').values
print(x133[0][7])
y133 = 1.31
columns = list(df.columns)
action_old = [float(df.values[:, i-6]) for i in range(6)]
max_step = 0
for i in range(6):
	step = int(abs(action_old[i] - params[i])/delta[i])
	if step > max_step:
		max_step = step
loss = []
s = []
t_action = action_old
imgX = []
for i in range(max_step):
	print('---{}---'.format(i))
	for j in range(6):
		if abs(t_action[j] - params[j]) > delta[j]:
			if (t_action[j] - params[j]) > delta[j]:
				t_action[j] -= delta[j]
			else:
				t_action[j] += delta[j]
	#print(t_action)
	normal_action = [0, 0, 0, 0, 0, 0]
	normal_action[-6] = (t_action[-6]-2.3859664)/(2.607782-2.3859664)
	normal_action[-5] = (t_action[-5]-0.3016754)/(100.81921-0.3016754)
	normal_action[-4] = (t_action[-4]-2.8080836)/(83.222635-2.8080836)
	normal_action[-3] = (t_action[-3]-3.6843995)/(64.396493-3.6843995)
	normal_action[-2] = (t_action[-2]-334.99402)/(457.82386-334.99402)
	normal_action[-1] = (t_action[-1]-0.4305181)/(0.6833051-0.4305181)
	for j in range(6):
		x133[0][j-6] = normal_action[j]
	x133[0][7] = (i/max_step) * (0.3333)
	s.append(x133[0][7] * 5.4 + 3.2)
	print(clf.predict(x133))
	loss.append(float(clf.predict(x133)))
	imgX.append([t_action[-4], t_action[-3], t_action[-2]])

plt.plot(s)
plt.title('133 产品硫含量随时间变化')
plt.grid()
plt.text(5, 3.2, (0, 3.2), color='r')
plt.plot(0, 3.2, '.', color='r')
plt.text(75, 5, (86, 5), color='r')
plt.plot(86, 5, '.', color='r')
plt.xlabel('Time step')
plt.ylabel('产品硫含量（μg/g）')

plt.show()
plt.plot(loss)
plt.title('133 RON损失值随时间变化')
plt.grid()
plt.text(2, 1.275, (0, 1.28), color='r')
plt.text(77, 0.87, (86, 0.88), color='r')
plt.plot(0, 1.28, '.', color='r')
plt.plot(86, 0.885, '.', color='r')
plt.xlabel('Time step')
plt.ylabel('RON loss')
plt.show()
print(imgX)

def plot_embedding_3d_xs(x, y, title=None, c1=2, c2=2, c3=200):
	x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
	x = (x - x_min)/(x_max - x_min)
	fig = plt.figure()
	y_normal = (y-np.min(y))/(np.max(y)-np.min(y))
	ax = fig.add_subplot(1,1,1, projection='3d')
	cmap = cmx.get_cmap('rainbow', 20)
	for i in range(x.shape[0]):
		if i%10==0:
			if i==0:
				ax.text(x[i,0], x[i,1], x[i,2], 'start:'+str(round(y[i], 2)), color=cmap(1-y_normal[i]), fontdict={'weight':'bold', 'size':14})
			elif abs(i-x.shape[0]) <= 10:
				pass
			else:
				ax.text(x[i,0], x[i,1], x[i,2], str(round(y[i], 2)), color=cmap(1-y_normal[i]), fontdict={'weight':'bold', 'size':14})				
		elif i == x.shape[0]-1:
			ax.text(x[i,0], x[i,1], x[i,2], 'end:'+str(round(y[i], 2)), color=cmap(1-y_normal[i]), fontdict={'weight':'bold', 'size':14})				
		ax.text(x[i,0], x[i,1], x[i,2], 'x', color=cmap(1-y_normal[i]), fontdict={'size':10})
	ax.set_xlabel('S-ZORB.TE_7508B.DACA')
	ax.set_ylabel('S-ZORB.TE_7108B.DACA')
	ax.set_zlabel('S-ZORB.TC_1607.DACA')
	if title is not None:
		plt.title(title)
		plt.show()

def plot_embedding_3d(x, y, title=None, c1=2, c2=2, c3=200):
	x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
	x = (x - x_min)/(x_max - x_min)
	fig = plt.figure()
	y_normal = (y-np.min(y))/(np.max(y)-np.min(y))
	ax = fig.add_subplot(1,1,1, projection='3d')
	cmap = cmx.get_cmap('rainbow', 20)
	for i in range(x.shape[0]):
		if i == 0:
			ax.text(x[i,0], x[i,1], x[i,2], str(round(y[i], 2))+'_start', color=cmap(1-y_normal[i]), fontdict={'weight':'bold', 'size':14})
		elif i == x.shape[0]-1:
			ax.text(x[i,0], x[i,1], x[i,2], str(round(y[i], 2))+'_end', color=cmap(1-y_normal[i]), fontdict={'weight':'bold', 'size':14})
		else:
			ax.text(x[i,0], x[i,1], x[i,2], str(round(y[i], 2)), color=cmap(1-y_normal[i]), fontdict={'weight':'bold', 'size':14})				
	ax.set_xlabel('S-ZORB.TE_7508B.DACA')
	ax.set_ylabel('S-ZORB.TE_7108B.DACA')
	ax.set_zlabel('S-ZORB.TC_1607.DACA')
	if title is not None:
		plt.title(title)
		plt.show()

plot_embedding_3d_xs(imgX, s, '133 产品硫含量与三个操作变量变化关系', 2, 2, 200)
plot_embedding_3d_xs(imgX, loss, '133 RON损失与三个操作变量变化关系', 3, 3, 120)
plot_embedding_3d(imgX, s, '133 产品硫含量与三个操作变量', 2, 2, 200)
plot_embedding_3d(imgX, loss, '133 RON损失与三个操作变量', 3, 3, 120)









