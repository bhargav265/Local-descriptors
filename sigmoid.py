import math

def sigmoid(x):
	return (1/(1+math.exp(-x)))

def f(x,y):
	return (x-y+54)+(x-54)*(y-54)

x =[54,47,28,52]
y = [114,60,55,72]

for i in range(4):
	print(sigmoid(f(x[i],y[i])))