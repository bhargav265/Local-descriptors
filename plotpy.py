import numpy as np 
import matplotlib.pyplot as plt
import math

def f1(x):
	if x>0:
		return 1
	else :
		return 0
def f2(x):
	return [1/(1+math.exp(x))]
def f3(x):
	return 1/(1+math.exp(-x))


x = np.arange(-20, 20., 0.01)
y1 = np.zeros(len(x))
y2 = np.zeros(len(x))

for i in range(0,len(x)):
	y1[i] = 1/(1+math.exp(x[i]))
	y2[i] = 1/(1+math.exp(-x[i]))

plt.plot(x, np.piecewise(x, [x  > 0, x <= 0], [0, 1]), 'r', label = 'CS-LDP threshold function')
plt.plot(x,y1, label='Modified sigmoid function')
plt.plot(x,y2, label = 'Sigmoid function')
plt.legend(loc="lower right")

plt.axis([-5, 5, -2, 3])

plt.show()