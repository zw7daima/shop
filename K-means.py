#-*- coding: utf-8 -*-
#使用K-Means算法聚类消费行为特征数据

import numpy as np
import pandas as pd
#参数初始化
inputfile = 'data_dropna2.xlsx' #销量及其他属性数据
k = 3 #聚类的类别
threshold = 2 #离散点阈值
iteration = 500 #聚类最大循环次数
data = pd.read_excel(inputfile) #读取数据
data_zs = 1.0*(data - data.mean())/data.std() #标准差数据标准化(0均值归一化)

from sklearn.cluster import KMeans
model = KMeans(n_clusters = k, n_jobs = 4, max_iter = iteration) #分为k类，并发数4
model.fit(data_zs) #开始聚类

#标准化数据及其类别
r = pd.concat([data_zs, pd.Series(model.labels_, index = data.index)], axis = 1)  #每个样本对应的类别
r.columns = list(data.columns) + [u'聚类类别'] #重命名表头

norm = []
for i in range(k): #逐一处理
  norm_tmp = r[['氨氮（毫克/升）','水质类别','pH值','溶解氧（毫克/升）','化学需氧量（毫克/升）','总磷（毫克/升）']][r[u'聚类类别'] == i]-model.cluster_centers_[i]
  #简化为r[a][b==i] - c[i]
  norm_tmp = norm_tmp.apply(np.linalg.norm, axis = 1) #求出绝对距离
  norm.append(norm_tmp/norm_tmp.median()) #求相对距离并添加

norm = pd.concat(norm) #合并
# print(norm)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
norm[norm <= threshold].plot(style = 'go') #正常点

discrete_points = norm[norm > threshold] #离群点
# print(norm[norm <= threshold])
discrete_points.plot(style = 'ro')

plt.xlabel(u'编号')
plt.ylabel(u'相对距离')
plt.show()


# 输出检测到的异常点和非异常点的文件分别为 normal.xlsx和abnormai.xlsx
data_zs = pd.DataFrame(data_zs).reset_index().drop(['index'],axis=1)

# 正常点
normal = data_zs[norm <= threshold]
# 不正常点
abnormal = data_zs[norm > threshold]

normal.to_excel("normal.xlsx")
abnormal.to_excel("abnormai.xlsx")

# 得到后续研究需要分类的数据
class_normal = normal.copy()
class_abnormal = abnormal.copy()
class_normal['类别'] = 0
class_abnormal['类别'] = 1

class_data = pd.concat([class_normal,class_abnormal],axis=0)

class_data.to_excel("class_data.xlsx")


# print(normal)
# print(abnormal)

# 在源数据通过PCA降维到两维的离群点检测可视化
from sklearn.decomposition import PCA
pca = PCA(n_components=3)# 保证降维后的数据保持90%的信息
pca.fit(data_zs)
# print(data_zs)
PCA_data_zs = pd.DataFrame(pca.transform(data_zs)).reset_index().drop(['index'],axis=1)
# print(PCA_data_zs)
# 正常点
normal = PCA_data_zs[norm <= threshold]
# 不正常点
abnormal = PCA_data_zs[norm > threshold]

# 画图

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y, z = normal[0], normal[1], normal[2]
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(x, y, z, c='g')  # 绘制数据点
# ax.scatter(x[10:20], y[10:20], z[10:20], c='y')
# ax.scatter(x[30:40], y[30:40], z[30:40], c='g')

x, y, z = abnormal[0], abnormal[1], abnormal[2]
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(x, y, z, c='r')  # 绘制数据点

ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()
# 参数说明：
# x指定散点图水平轴表示的变量,y指定散点图垂直轴表示的变量
# kind=scatter表示绘制的是散点图,s指定散点的面积,c指定散点的颜色
# marker指定散点的形状,ax指定散点图绘制的子图位置


print(normal)
print(abnormal)


