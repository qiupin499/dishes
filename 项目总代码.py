#载入模块
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#任务1.1：载入数据
data = pd.read_csv('meal_order_detail.csv')
infodata = pd.read_csv('meal_order_info.csv')
print('原detail表数据形状：',data.shape)
data_wash = data

#任务1.2：去除菜品名称中的空格和回车
data_wash = data_wash.replace('\n','', regex=True)
data_wash.loc[:,'dishes_name']=data_wash.loc[:,'dishes_name'].str.strip()

#任务1.3：各个菜品的热销度评分
#在已清洗数据中除去白饭
data_wash = data_wash.drop(index=(data_wash.loc[(data_wash['dishes_name']=='白饭/小碗')].index))
data_wash = data_wash.drop(index=(data_wash.loc[(data_wash['dishes_name']=='白饭/大碗')].index))
#统计各个菜品的销售数量
dish_name = data_wash['dishes_name']
dish_name = list(dish_name.drop_duplicates())
dish_num = {}
for i in dish_name:
    dish_num[i] = 0
for i in dish_name:
    for j,h in zip(data_wash['dishes_name'],data_wash['counts']):
        if i==j:
            dish_num[i] = dish_num[i]+h
dish_num_sorted = sorted(dish_num.items(), key=lambda x: x[1],reverse=True)
print('各个菜品的销售数量分别为：',dish_num_sorted)
#创建菜品热销度字典
min_sale = dish_num_sorted[-1][1] #最小销量
max_sale = dish_num_sorted[0][1] #最大销量
dish_sale = {}
for i in dish_name:
    dish_sale[i] = (dish_num[i]-min_sale)/(max_sale-min_sale)
dish_sale_sorted = sorted(dish_sale.items(), key=lambda x: x[1],reverse=True)
print('菜品热销度分别为：',dish_sale_sorted)

#任务1.4：绘制条形图展示热销top10
#找到热销top10及其销量
dish_salenum_top10 = dish_num_sorted[0:10]
#绘制条形图
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.rcParams['font.sans-serif']=['SimHei'] #这行代码用于显示中文，'SimHei'就是黑体
dish_salenum_top10 = pd.DataFrame(dish_salenum_top10,columns=['菜品','销量'])
dish_salenum_top10.plot(y='销量',x='菜品',kind='bar',grid=False,title='热销top10及其销量')
plt.show()

#任务2.1：统计各种订单状态所占比例
order_status_counts = infodata['order_status'].value_counts()
print('各种订单状态所占比例（1为正常）：',order_status_counts/len(infodata))

#任务2.2：除去无效的订单，只保留正常的订单
infodata_wash = infodata.drop(index=(infodata.loc[(infodata['order_status']==0)].index))
infodata_wash = infodata_wash.drop(index=(infodata_wash.loc[(infodata_wash['order_status']==2)].index))
#数据清洗：将info表中有而detail表中没有，或info表中没有而detail表中有的订单除去
infodata_wash.rename(columns={'info_id': 'order_id'}, inplace=True) #将两表的订单id名称统一
order_id = infodata_wash['order_id']
order_id = list(order_id.drop_duplicates())
for i in data_wash['order_id']:
    n = 0
    for j in order_id:
        if i==j:
            n=n+1
    if n==0:
        data_wash = data_wash.drop(index=(data_wash.loc[(data_wash['order_id'] == i)].index))
print('清洗后的detail表形状：',data_wash.shape)

#任务2.3：保留主要特征emp_id（用户id）和dishes_name（菜品名称）
data_wash_columns = list(data_wash.columns)
for i in data_wash_columns:
    if i!='emp_id' and i!='dishes_name':
        data_wash.drop(i,axis=1,inplace=True)
print('进一步清洗后（保留主要特征后）的detail表形状：',data_wash.shape)

#任务3.1：按用户id来划分训练集（70%）和测试集（30%）
#划分训练集用户和测试集用户
emp_id = data_wash['emp_id']
emp_id = list(emp_id.drop_duplicates())
test_emp_id = []
train_emp_id = []
np.random.seed(6) #设置随机种子
for i in emp_id:
    if np.random.randint(0,10)<3:
        test_emp_id.append(i)
    else:
        train_emp_id.append(i)
print('测试集用户数：',len(test_emp_id))
print('训练集用户数：',len(train_emp_id))
#划分出dataframe格式的训练集和测试集
test_data = data_wash
train_data = data_wash
for i in test_emp_id:
    train_data = train_data.drop(index=(train_data.loc[(train_data['emp_id']==i)].index))
for i in train_emp_id:
    test_data = test_data.drop(index=(test_data.loc[(test_data['emp_id']==i)].index))
print('测试集形状',test_data.shape)
print('训练集形状',train_data.shape)

#任务3.2：构建训练集用户-菜品二维矩阵
m = range(len(dish_name))
n = range(len(train_emp_id))
train_matrix = np.mat(np.zeros((len(dish_name),len(train_emp_id)))) #第一层按菜品名，排序与dish_name列表相同；第二层按用户id，排序与train_emp_id相同。
for i in m:
    for j in n:
        if train_data[(train_data.loc[:,'dishes_name']==dish_name[i]) & (train_data.loc[:,'emp_id']==train_emp_id[j])].empty==False:
            train_matrix[i,j]=1 #该循环消耗时间很长
print('训练集用户-菜品二维矩阵为:',train_matrix)

#模型构建：采用基于物品的协同过滤算法
#任务4.1：采用Jaccard相似度方法计算相似度矩阵
similar_matrix = np.mat(np.zeros((len(dish_name),len(dish_name)))) #菜品与菜品之间的Jaccard相似度作为值，菜品排序与dish_name相同。
for i in m:
    for h in m:
        cross = 0
        union = 0
        for j in n:
            if train_matrix[i,j]==1 and train_matrix[h,j]==1:
                cross=cross+1
            if train_matrix[i,j]==1 or train_matrix[h,j]==1:
                union=union+1
        similar_matrix[i,h]=cross/union
print('菜品相似度矩阵为：',similar_matrix)

#任务4.2：针对目标客户生成推荐列表，这里以测试集用户为目标客户。若A已点且只点了b、c，则A用户对b菜品的兴趣分为(1*b与b的相似度+1*b与c的相似度)
test_emp_interest_matrix = np.mat(np.zeros((len(test_emp_id),len(dish_name)))) #创建测试集用户的菜品推荐矩阵， #第一层按用户id，排序与test_emp_id列表相同；第二层按菜品名，排序与dish_name相同。
for j in range(len(test_emp_id)):
    test_emp_dishdata = test_data[(test_data.loc[:, 'emp_id'] == test_emp_id[j])]  # 将当前客户点的所有菜品检索出
    test_emp_dishdata = list(test_emp_dishdata['dishes_name'])
    for i in m:
        interest_score = 0
        for h in test_emp_dishdata:
            index = dish_name.index(h)
            interest_score = interest_score + 1*similar_matrix[i,index]
        test_emp_interest_matrix[j,i] = interest_score
print('测试集用户的推荐列表：',test_emp_interest_matrix)

#任务5.1：构建测试集客户的已点菜品字典
test_emp_data = {}
for j in range(len(test_emp_id)):
    test_emp_dishdata = test_data[(test_data.loc[:, 'emp_id'] == test_emp_id[j])]  # 将当前客户点的所有菜品检索出
    test_emp_dishdata = list(test_emp_dishdata['dishes_name'])
    test_emp_data[test_emp_id[j]] = test_emp_dishdata
print('测试集客户的已点菜品字典',test_emp_data)

#任务5.2：构建评价指标，分析模型推荐效果
test_emp_accuracy = {} #构建测试集用户的推荐准确度字典
for j in range(len(test_emp_id)):
    interest_len = len(test_emp_data[test_emp_id[j]]) #设置系统应推荐给当前客户的菜品种类与该客户的实际点菜种类相等。
    true_len = 0
    a = test_emp_interest_matrix[j,:].tolist()[0] #当前用户的菜品推荐列表
    b = sorted(a,reverse=True)
    for i in b[0:interest_len]: #推荐列表中兴趣分排名靠前的菜品（只取排前面的系统应推荐量的名次）即为系统推荐菜品
        if dish_name[a.index(i)] in test_emp_data[test_emp_id[j]]:
            true_len = true_len + 1 #当前用户的实际已点菜品中存在推荐的菜品，每有一个，准确度+1
    test_emp_accuracy[test_emp_id[j]] = true_len/interest_len
print('测试集用户的推荐准确度字典',test_emp_accuracy)
all_accuracy = 0 #计算模型整体准确度，为所有测试集用户推荐准确度的均值
for j in range(len(test_emp_id)):
    all_accuracy = all_accuracy + test_emp_accuracy[test_emp_id[j]]
all_accuracy = all_accuracy/len(test_emp_id)
print('模型总体准确率为：',all_accuracy)

