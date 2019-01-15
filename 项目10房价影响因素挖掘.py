import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from bokeh.plotting import figure,show,output_file
from bokeh.models import ColumnDataSource
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  
# Matplotlib中设置字体-黑体，解决Matplotlib中文乱码问题
plt.rcParams['axes.unicode_minus'] = False    
# 解决Matplotlib坐标轴负号'-'显示为方块的问题
sns.set(font_scale=1.5,font='SimHei')
#读取数据
os.chdir('D:\\数据分析项目\\项目10房价影响因素挖掘')
df1 = pd.read_csv('house_rent.csv',engine = 'python')
df2 = pd.read_csv('house_sell.csv',engine = 'python')
'''
part1 查看租售比分布情况
'''

#数据一去除缺失值
df1.dropna(inplace = True)
df2.dropna(inplace = True)
#计算每平米每月的租金价格
df1['rent_area'] = df1['price'] / df1['area']
df_rent = df1[['community','rent_area','lng','lat']].groupby('community').mean()
df_sell = df2[['property_name','average_price','lng','lat']].groupby('property_name').mean()
df_rent.reset_index(inplace = True)
df_sell.reset_index(inplace = True)
df3 = pd.merge(df_rent,df_sell,left_on = 'community',right_on = 'property_name')
df3 = df3[['community','rent_area','average_price','lng_x','lat_x']]
df3.columns = ['community','rent_area','sell_area','lng','lat']
df3['r/s'] = df3['sell_area'] / df3['rent_area']
#绘制租售比直方图
plt.figure(figsize = (10,4))
sns.distplot(df3['r/s'],bins = 100,color = 'g',kde = False,
             hist_kws = dict(edgecolor = 'w',linewidth = 1))
plt.title('房屋租售比直方图')
plt.xlabel('租售比')
plt.figure(figsize= (10,4))
sns.boxplot(x = df3['r/s'],color = '#D94F56',width = 0.5,linewidth = 1)
plt.xlabel('租售比')
plt.title('房屋租售比箱型图')
#导出数据
write = pd.ExcelWriter('租售比.xlsx')
df3.to_excel(write,'sheet1')
write.save()

data = pd.read_csv('datapoint .csv',engine = 'python')
data.fillna(0,inplace = True)
def f_nor(data,col):
    return((data[col] - data[col].min()) / (data[col].max()-data[col].min()))
data['人口密度指数'] = f_nor(data,'Z')
data['道路密度指数'] = f_nor(data,'长度')
data['餐饮密度指数'] = f_nor(data,'人均消费_')
#绘制人口密度和房价的散点图
data_nor = data[['人口密度指数','道路密度指数','餐饮密度指数','sell_area_']]
data_nor['离市中心的距离'] = ((data['lng'] - 353508.848122)**2 + (data['lat'] - 3456140.926976) **2)**0.5
data_nor = data_nor[data_nor['sell_area_'] > 0]
data_nor.reset_index(inplace =True,drop = True)
plt.figure(figsize = (10,4))
plt.scatter(x =data_nor['人口密度指数'],y = data_nor['sell_area_'],alpha= 0.8 ,s = 3)
plt.title('人口密度和房屋均价的关系')
plt.xlabel('人口密度指数')
plt.ylabel('房屋每平米均价')
''' 可以看出人口密度和房价之间的关系不太明显'''
#绘制道路密度和房价的散点图
plt.figure(figsize = (10,4))
plt.scatter(x =data_nor['道路密度指数'],y = data_nor['sell_area_'],alpha= 0.8 ,s = 3)
plt.title('道路密度和房屋均价的关系')
plt.xlabel('道路密度指数')
plt.ylabel('房屋每平米均价')
''' 可以看出道路密度和房价之间的有较为明显的线性关系'''
#绘制餐饮密度和房价之间的散点图
plt.figure(figsize = (10,4))
plt.scatter(x =data_nor['餐饮密度指数'],y = data_nor['sell_area_'],alpha= 0.8 ,s = 3)
plt.title('餐饮密度和房屋均价的关系')
plt.xlabel('餐饮密度指数')
plt.ylabel('房屋每平米均价')
''' 可以看出道路密度和房价之间基本没有线性关系'''
#计算各个点离市中心的距离并绘图

plt.figure(figsize = (10,5))
plt.scatter(x =data_nor['离市中心的距离'],y = data_nor['sell_area_'],alpha= 0.8 ,s = 3)
plt.title('离市中心距离和房屋均价的关系')
plt.xlabel('离市中心距离')
plt.ylabel('房屋每平米均价')
'''可以看出距离市中心的距离和房价有着强负相关的关系'''
'''房价跟与市中心的距离有一定的离散程度，越靠近市中心离散越大，越远离市中心离散程度越小'''
#用correct（）函数得出房价和以上四种因素的皮尔逊相关系数
data_nor.corr()
print(data_nor.corr().loc['sell_area_'])

'''
房价与市中心的距离之间离散程度的影响关系
'''
#将10km作为一个区间 计算每个指标的相关系数
rkmd = []
dlmd = []
cymd = []
center = []
juli = []
m = 10000
j = 0
while 1:
    if j < 60000:
        data_re = data_nor[data_nor['离市中心的距离'] < (m+j)]
        x = data_re.corr().loc['sell_area_']
        rkmd.append(x[0])
        dlmd.append(x[1])
        cymd.append(x[2])
        center.append(x[4])
        j += 10000
        juli.append(j)
    else:
        break
data_bokeh = pd.DataFrame({'rkmd':rkmd,'dlmd':dlmd,
                           'cymd':cymd,'center':center,'juli':juli})
from bokeh.models import HoverTool
source = ColumnDataSource(data = data_bokeh)
hover = HoverTool(tooltips = [('离市中心距离','@juli'),
                              ('人口密度相关系数','@rkmd'),
                              ('道路密度相关系数','@dlmd'),
                              ('餐饮密度相关系数','@cymd'),
                              ('市中心距离相关系数','@center')
                                ])
p = figure(plot_width = 800,plot_height = 350,
           title = '随着离市中心距离的变化 各指标相关系数的变化趋势',
           tools=[hover,'box_select,reset,xwheel_zoom,pan,crosshair'])
#p.multi_line(['juli','juli','juli','juli'],['rkmd','dlmd','cymd','center'],source = source)
#p.line('juli','rkmd',source = source,color = )
#show(p)
colors = ['#E71D36','#FF9F1C','#2EC4B6','#011627']
legend = ['人口密度相关系数','道路密度相关系数','餐饮密度相关系数','市中心距离相关系数']
for i,j,k in zip(data_bokeh.columns.tolist()[:4],colors,legend):
    p.line('juli',i,source = source,color = j,line_width = 2,
           line_dash = [6,4],alpha = 0.6,legend = k)
    p.legend.location = 'center_right'
    p.circle('juli',i,source = source,color = j,size = 6)
show(p)
'''
结论:
  1、从上图可以看出，房价跟人口密度、道路密度、离市中心的距离有着较强的相关性，尤其是离市中心的的距离，
     越远离市中心房价越低且房价的离散程度较弱，越靠近市中心房价越高且房价的离散程度较高，但房价跟餐饮的
     密集程度相关性较弱
  2、在离市中心3km-40km这个区段 各指标相关性出现了较为缓慢的变化，
  这正是上海中心城区和郊区的分界 → 上海房价市场的“中心城区-郊区”分化特征
  3、中心城区的房产市场对指标因素的影响更加敏锐，而郊区则更迟钝 → 越靠近市中心，影响因素越复杂
'''













