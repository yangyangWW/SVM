import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm

#获取数据源
data = pd.read_csv('C:/Users/34563/Desktop/WS_TestsStudy/WS_Iterms/SVM/breast_cancer_data-master/data.csv')
#数据探索
#将数据中的列全部显示出来
pd.set_option('display.max_columns',None,'display.max_rows',10)
print(data.columns)
print(data.head(5))
print('---'*30)
print(data.describe)
#数据清洗：没有缺失值，将数据列features分开,去掉无用的列数据，将数据中的字符列换为1/0表示，
features_mean = list(data.columns[2:12])
features_se = list(data.columns[13:22])
features_worst = list(data.columns[22:32])

'''features显示
display(features_mean,features_se,features_worst)
#['radius_mean',
 'texture_mean',
 'perimeter_mean',
 'area_mean',
 'smoothness_mean',
 'compactness_mean',
 'concavity_mean',
 'concave points_mean',
 'symmetry_mean',
 'fractal_dimension_mean']
['texture_se',
 'perimeter_se',
 'area_se',
 'smoothness_se',
 'compactness_se',
 'concavity_se',
 'concave points_se',
 'symmetry_se',
 'fractal_dimension_se']
['radius_worst',
 'texture_worst',
 'perimeter_worst',
 'area_worst',
 'smoothness_worst',
 'compactness_worst',
 'concavity_worst',
 'concave points_worst',
 'symmetry_worst',
 'fractal_dimension_worst']
'''

data.drop('id',axis = 1, inplace = True) #去掉对分析无用的id列

data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
print(data['diagnosis'].head(5))   #输出看一下这一列数据是否改变

#将肿瘤诊断结果可视化
sns.countplot(data['diagnosis'],label = 'Count')
plt.show
#热力图呈现features_mean字段之间的相关性
corr = data[features_mean].corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True)  #annot=True会显示每个方格的数据
plt.show()
#特征选择,特征字段保留的越多预测结果会准确一些，但是计算量会大，因此要找一个平衡点，使得方案最优
features_remain = ['radius_mean','texture_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean']
#抽取30%数据作为测试集
train,test = train_test_split(data,test_size=0.33)
#抽取特征选择的数值和判断结果分别作为特征属性和分类标签
train_x = train[features_remain]
train_y = train['diagnosis']
test_x = test[features_remain]
test_y = test['diagnosis']

#在训练之前对数据进行标准正态规范化，保证每个特征维度的数值均值为0，方差为1，
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()  
train_x = ss.fit_transform(train_x)
test_x = ss.fit_transform(test_x)
#SVM做训练和预测
#创建分类器
model = svm.SVC()   #非线性分类器
model = svm.LinearSVC  #线性分类器

#拟合
model.fit(train_x,train_y)
#预测并计算准确率
predict_y = model.predict(test_x)

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(predict_y,test_y)
print('SVM预测准确率：',acc_score)
#SVM预测准确率： 0.9308510638297872   #features_remain保留了8个字段，model = svm.SVC，相关性大于0.98的合并只取一个字段
#SVM预测准确率： 0.9414893617021277   #features_remain保留了8个字段，model = svm.LinearSVC
#SVM预测准确率： 0.9414893617021277   #features_remain保留了7个字段，model = svm.SVC，相关性大于0.9的合并只取一个字段
#SVM预测准确率： 0.9361702127659575   #features_remain保留了7个字段，model = svm.LinearSVC
#小结：由以上测试结果来看，相关性大于0.9的几个相关字段可以只取一个作为特征属性，两种svm模型拟合结果相差不算大。当然如果取全部特征字段进行拟合效果会更好。














