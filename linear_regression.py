#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys, os, math
from GradientDescent import GradientDescent
from datetime import timedelta
from collections import OrderedDict

### preproc data
#对range(24)各项进行str操作，#返回列表['0', '1', '2', '3', ...,'23'] 
      
colnames = ['Date',"Site","Item"]+list(map(str,range(24)))
# 读取CSV文件，header参数指定从第几行开始生成，且将header行的数据作为列的name（键），header行以前的数据将不会处理。
# 取值为None表示csv中行不做为列的name（键），取值为0表示将csv的第0行作为列的name。
# 如果没有传递参数names那么header默认为0；如果传递参数names，那么header默认为None。
# 此处没有传递参数names那么header默认为0，表示将csv的第0行作为列的name
# 需要忽略的行数（从文件开始处算起），或需要跳过的行号列表（从0开始）。
df = pd.read_csv("./data/train.csv",names=colnames,skiprows=1)

# remove column "Site"
#数据df中选择保留'Date',"Item"以及range列
df = df.loc[:,['Date',"Item"]+list(map(str,range(24)))]

# melt "Hour" to column
# id_vars:不需要被转换的列名。value_vars:需要转换的列名（变换后一行只有一个值）
df = pd.melt(df, id_vars=['Date','Item'], value_vars = map(str,range(24)), var_name='Hour', value_name='Value')

# generate "Datetime"获取指定的时间和日期
df["Datetime"] = pd.to_datetime(df.Date + " " + df.Hour + ":00:00")
#数据df中选择保留'Datetime',"Item"以及Value列
df = df.loc[:,['Datetime',"Item","Value"]]

# replace NR to 0 按条件取数据并替换
df.loc[df.Value=="NR","Value"] = 0

# change "Value" type
df["Value"] = df["Value"].astype(float)

# pivot 'Item' to columns
#透视表比较智能，通过将“Datetime”(index)列进行对应分组，来实现数据聚合和总结
#变量“columns（列）”提供一种额外的方法来分割你所关心的实际值。聚合函数aggfunc被应用到了变量“values”中你所列举的项目上。
df = df.pivot_table(values='Value', index='Datetime', columns='Item', aggfunc='sum')

### obtain training set and validation set
#收集12月分数据和非12月份数据
df_12m     = df.loc[df.index.month==12,:]
df_not_12m = df.loc[df.index.month!=12,:]

def gen_regression_form(df):
    data = OrderedDict()
    #得到df所有的列标签使用tolist()函数转化为list 5760x18->
    item_list = df.columns.tolist()
    datetime_list = df.index
    for i in range(9):
        for item in item_list:
            data['{:02d}h__{}'.format(i+1,item)]=[]#数字补零 (填充左边, 宽度为2)
    data['10h__PM2.5'] = []

    d1h = timedelta(hours=1)#1个小时的时间单位
    for m in pd.unique(datetime_list.month):#pd.unique(Series)获取Series中元素的唯一值（即去掉重复的）
        for timestamp in (df.loc[df.index.month==m,:]).index:
            start = timestamp
            end   = timestamp + 9*d1h
            sub_df = df.loc[(start <= df.index) & (df.index <= end),:]#从df中提取满足时间要求子数据
            if sub_df.shape[0] == 10:#.shape[0] 为list第二维的长度
                for i in range(9):
                    for item in item_list:
                        data['{:02d}h__{}'.format(i+1,item)].append(
                            sub_df.loc[timestamp+i*d1h,item] )
                data['10h__PM2.5'].append(sub_df.loc[timestamp+9*d1h,'PM2.5'])
    
    return pd.DataFrame(data)

path_valid_data = './valid_data.csv'
if os.path.isfile(path_valid_data):
    valid_data = pd.read_csv(path_valid_data)
else:
    valid_data = gen_regression_form(df_12m)#480x18->471x163
    valid_data.to_csv(path_valid_data,index=None)

path_train_data = './train_data.csv'
if os.path.isfile(path_train_data):
    train_data = pd.read_csv(path_train_data)
else:
    train_data = gen_regression_form(df_not_12m)
    train_data.to_csv(path_train_data,index=None)

train_X = np.array(train_data.loc[:,train_data.columns!='10h__PM2.5'])
train_y = np.array(train_data.loc[:,'10h__PM2.5'])

valid_X = np.array(valid_data.loc[:,valid_data.columns!='10h__PM2.5'])
valid_y = np.array(valid_data.loc[:,'10h__PM2.5'])

# record the order of columns
colname_X = (train_data.loc[:,train_data.columns!='10h__PM2.5']).columns

### gradient descent
gd = GradientDescent()


gd.train_by_pseudo_inverse(train_X,train_y,alpha=0.5,validate_data = (valid_X,valid_y))
init_wt = gd.wt
init_b  = gd.b

gd.train(train_X,train_y,epoch=10,rate=0.000001,batch=100,alpha=0.00000001,
    init_wt=np.array(init_wt),init_b=init_b,
    validate_data = (valid_X,valid_y))

### testing
col_names = ['ID','Item']+list(map(lambda x:'{:02d}h'.format(x),range(1,10)))
test = pd.read_csv('./data/test_X.csv', names = col_names, header=None )

# record ordfer of test.ID
id_test = test.ID

# replace NR to 0
for col in map(lambda x:'{:02d}h'.format(x),range(1,10)):
    test.loc[(test.Item=='RAINFALL')&(test[col]=='NR'),col] = 0

# ['ID','Item','Hour','Value'] form
test =  test.pivot_table( index=['ID','Item'], aggfunc='sum')
test = test.stack()
test = test.reset_index()
test.columns = ['ID','Item','Hour','Value']

# combine 'Hour' and 'Item' to 'Col'
test['Col'] = test.Hour + "__" + test.Item
test = test[['ID','Col','Value']]

# pivot 'Col' to columns
test = test.pivot_table(values='Value',index='ID',columns='Col', aggfunc='sum').reset_index()
test.name = ''

# re-order
test['ID_Num'] = test.ID.str.replace('id_','').astype('int')
test = test.sort_values(by='ID_Num')
test = test.reset_index(drop=True)

# predict
X_test = np.array(test[colname_X],dtype='float64')
test['Predict'] = gd.predict(X_test)

# output
test[['ID','Predict']].to_csv('linear_regression.csv',header=None,index=None)
