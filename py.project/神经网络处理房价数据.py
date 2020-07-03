#import…as是一种导入库的方法
import numpy as np
# 定义存储输入数据x和目标数据y的数组
x, y = [], []
for sample in open("E:\_Data\prices3.txt", "r"):
    x.append([float(sample.split(",")[0]),float(sample.split(",")[1])])
    y.append(float(sample.split(",")[2]))
x, y = np.array(x), np.array(y)


# sigmoid激活函数定义
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# 定义损失函数
def get_cost(input_x, input_y):
    return 0.5 * ((input_x - input_y) ** 2)
#计算采用get_model 模型处理输入input_x得到的预测结果，与真实结果input_y，损失函数结果。

class NeuralNetwork():       
    def __init__(self):
        
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        
    def forward(self,x):
        outh1=sigmoid(self.w1 * x[0] + self.w2 * x[1] +self.b1)
        outh2=sigmoid(self.w3 * x[0] + self.w4 * x[1] +self.b2)
        outh3=sigmoid(self.w5 * outh1 + self.w6 * outh2 + self.b3)
        return outh3
    
    def train(self, data, y):
        learn_rate = 0.1
        epochs = 100 # number of times to loop through the entire dataset
        for epoch in range(epochs):
            for x, y_trues in zip(data, y):
               
                # 神经网络的前向（前馈）计算
                h1 = self.w1 * x[0] + self.w2 * x[1] +self.b1
                h2 = self.w3 * x[0] + self.w4 * x[1] +self.b2
                outh1=sigmoid(h1)
                outh2=sigmoid(h2)
                h3 = self.w5 * outh1 + self.w6 * outh2 +self.b3
                outh3=sigmoid(h3)
                
                #梯度计算式子
                d_E_outh3= (outh3 - y_trues)
                d_outh3_h3= outh3 * (1-outh3)
                d_outh1_h1= outh1 * (1-outh1)
                d_outh2_h2= outh2 * (1-outh2)
               
                #反向传播，调整权值
                self.w5 -=learn_rate *d_E_outh3*d_outh3_h3*outh1
                self.w6 -=learn_rate *d_E_outh3*d_outh3_h3*outh2
                self.b3 -=learn_rate *d_E_outh3*d_outh3_h3

                self.w1 -=learn_rate *d_E_outh3*d_outh3_h3*self.w5 * d_outh1_h1 * x[0]
                self.w2 -=learn_rate *d_E_outh3*d_outh3_h3*self.w5 * d_outh1_h1 * x[1]
                self.b1 -=learn_rate *d_E_outh3*d_outh3_h3*self.w5 * d_outh1_h1
                self.w3 -=learn_rate *d_E_outh3*d_outh3_h3*self.w6 * d_outh2_h2 * x[0]
                self.w4 -=learn_rate *d_E_outh3*d_outh3_h3*self.w6 * d_outh2_h2 * x[1]
                self.b2 -=learn_rate *d_E_outh3*d_outh3_h3*self.w6 * d_outh2_h2
            
            if epoch % 10 == 0:
                loss = 0
                for i, yy in zip(data, y):
                    y_preds = self.forward(i)
                    loss+= get_cost(y_preds,yy)
                print("Epoch %d loss: %.3f", (epoch, loss))

network = NeuralNetwork()
network.train(x, y)




