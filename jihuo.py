# -*- coding : UTF-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
from math import exp

# set x's range
x = np.arange(-100, 100, 0.1)

y1 = 1 / (1 + math.e ** (-x))  # sigmoid
y11 = 1 / (2 + math.e ** (-x) + math.e ** (x))  # sigmoid的导数

y2 = (math.e ** (x) - math.e ** (-x)) / (math.e ** (x) + math.e ** (-x))  # tanh
y22 = 1 - y2 * y2  # tanh函数的导数

y3 = np.where(x < 0, 0, x)  # relu
y33 = np.where(x < 0, 0, 1)  # ReLU函数导数

y4 = np.where(x < 0, 0.01*x, x) #Leaky ReLU
y44 = np.where(x < 0, 0.01, 1)  #Leaky ReLU导数

y5 = np.log(np.exp(x) + 1)   #softplus
y55 = math.e ** (x) / (1+math.e ** (x))

y6 = np.where(x <= -3, 0,np.where(x >= 3,x,x*(x+3)/6))  #hardswish
y66 = np.where(x <= -3, 0, np.where(x >= 3, 1, (2*x)/6))

y7 = x * (np.exp(y5)-np.exp(-y5))/(np.exp(y5)+np.exp(-y5))


plt.xlim(-5, 5)
plt.ylim(-2, 2)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

# Draw pic
#plt.plot(x, y1, label='Sigmoid', linestyle="-", color="red")
#plt.plot(x, y11, label='Sigmoid derivative', linestyle="-", color="violet")

#plt.plot(x, y2, label='Tanh', linestyle="-", color="blue")
#plt.plot(x, y22, label='Tanh derivative', linestyle="-", color="violet")
#
#plt.plot(x, y3, label='Relu', linestyle="-", color="green")
#plt.plot(x, y33, label='Relu derivative', linestyle="-", color="violet")

#plt.plot(x, y4, label='Leaky ReLU', linestyle="-", color="olive")
#plt.plot(x, y44, label='Leaky ReLU derivative', linestyle="-", color="orangered")

#plt.plot(x, y5, label='Softplus', linestyle="-", color="dimgrey")
#plt.plot(x, y55, label='Softplus derivative', linestyle="-", color="rosybrown")

plt.plot(x, y6, label='Softplus', linestyle="-", color="purple")
plt.plot(x, y66, label='Softplus derivative', linestyle="-", color="deeppink")

#plt.plot(x, y7, label='Mish', linestyle="-", color="k")


# Title
plt.legend(['Sigmoid', 'Tanh', 'Relu', 'Leaky ReLU', 'Softplus','hardswish','Mish'])
#plt.legend(['Sigmoid', 'Sigmoid derivative'])  # y1 y11
#plt.legend(['Tanh', 'Tanh derivative'])  # y2 y22
plt.legend(['hardswish', 'hardswish derivative'])  # y3 y33
#plt.legend(['Leaky ReLU', 'Leaky ReLU derivative'])  # y4 y44
#plt.legend(['Mish', 'Mish derivative'])  # y5 y55


# plt.legend(['Sigmoid', 'Sigmoid derivative', 'Relu', 'Relu derivative', 'Tanh', 'Tanh derivative'])  # y3 y33
# plt.legend(loc='upper left')  # 将图例放在左上角

# save pic
plt.savefig('plot_test.png', dpi=100)
plt.savefig(r"./ReLU")

# show it!!
plt.show()

