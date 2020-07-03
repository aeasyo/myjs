#import…as是一种导入库的方法
import numpy as np
import matplotlib.pyplot as plt
# 定义存储输入数据x和目标数据y的数组
x, y = [], []
#遍历数据集，变量sample对应的是一个个样本，括号里为文件路径
for sample in open("D:\Pictures\prices.txt", "r"):
#采用python的split方法，具体方法参考网址https://www.runoob.com/python/att-string-split.html?ivk_sa=1023345p，split(",")这里是以逗号作为分隔符，将参数分别传给xx和yy。
    xx, yy = sample.split(",")
#用float()将字符串数据浮点化，其中append()方法用于在列表末尾添加新的对象，这里添加到输入数据x和目标数据y的数组后面。
    x.append(float(xx))
    y.append(float(yy))
#np.array与np.asarray功能是一样的，都是将输入转为矩阵格式，具体参考https://www.runoob.com/numpy/numpy-ndarray-object.html
x, y = np.array(x), np.array(y)
# 数据标准化
x = (x - x.mean()) / x.std()
# 将原始数据以散点图的形式画出


#在（-2，4）这个区间里取100个点作为画图的基础
x0 = np.linspace(-2, 4, 100)
# def 定义函数get_model，deg为参数，即x的最高次幂
def get_model(deg):
#这里用到np.polyfit()函数，是进行多项式拟合，求出求取一组p0-pn,使得loss，即损失函数的值最小。详情可参考网址https://www.cnblogs.com/maplethefox/p/11468296.html，根据拟合系数（即p0-pn）与自变量x求出拟合值,采用np.polyval()函数。接着返回的模型能够根据输入的x（默认是x0），返回相对应的预测的y。
    return lambda input_x=x0: np.polyval(np.polyfit(x, y, deg), input_x)
# 定义损失函数
def get_cost(deg, input_x, input_y):
    return 0.5 * ((get_model(deg)(input_x) - input_y) ** 2).sum()
#计算采用get_model 模型处理输入input_x得到的预测结果，与真实结果input_y，损失函数结果。
# 定义测试集
test_set = (1, 4, 10)
for d in test_set:
    print(get_cost(d, x, y))
# 画出相应的图像
plt.scatter(x, y, c="g", s=20)
for d in test_set:
    plt.plot(x0, get_model(d)(), label="degree = {}".format(d))
#用plt.plot绘制折线图
#将横轴、纵轴的范围分别限制在（-2，4）、（1e5, 8e5）
plt.xlim(-2, 4)
plt.ylim(1e5, 8e5)
#调用legend方法使得曲线显示对应的label
plt.legend()
plt.show()
