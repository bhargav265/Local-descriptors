import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
features1 = pd.read_csv("Original_Data/batch1.csv",header = None)

#features2 = pd.read_csv("Original_Data/batch2.csv",header = None)
#features3 = pd.read_csv("Original_Data/batch3.csv",header = None)
#features4 = pd.read_csv("Original_Data/batch4.csv",header = None)
#features5 = pd.read_csv("Original_Data/batch5.csv",header = None)
#features_test = pd.read_csv("Original_Data/test_batch.csv", header = None)

features1 = features1.values.astype(float)
'''
features2 = features2.values.astype(float)
features3 = features3.values.astype(float)
features4 = features4.values.astype(float)
features5 = features5.values.astype(float)
features_test = features_test.values.astype(float)
'''
'''
labels1 = features1[:,-1]
labels2 = features2[:,-1]
labels3 = features3[:,-1]
labels4 = features4[:,-1]
labels5 = features5[:,-1]

#print(features1.shape)
#print(features2.shape)
'''
#features = np.concatenate((features1,features2,features3,features4,features5),axis = 0)
#labels = np.concatenate((labels1,labels2,labels3,labels4,labels5),axis = 0)

#labels1 = labels1.reshape(10000,1)
#print(labels1.shape)
#print(features.shape)


#print("Features " + str(features.shape))
#print("Labels "+ str(labels1.shape))
#features = features.astype(np.int)
#print(features[0])
#print(features.shape)
LGP_features = np.zeros((50000,16))
LGP_features_test = np.zeros((10000,16))
#LBP_features_border = np.zeros((50000,16))
#LBP_features_center = np.zeros((50000,16))

def thresholded(pixels):
	out = []
	#mean = sum(pixels)/float(len(pixels))
	for a in pixels:
		if a>=0:
			out.append(1)
		else:
			out.append(0)
	
	return out

def get_pixel_else_0(center,l,idx,idy,default=0):
	try:
		return abs(l[idx,idy]-center)
	except IndexError:
		return default

def read_img():
	for i in range(1):
		img = features1[i,0:1024]
		print('processing ' + str(i))
		img = img.reshape((32,32))
		transformed_img = np.zeros((32,32))
		for x in range(0,len(img)):
			for y in range(0,len(img[0])):
				center        = img[x,y]
				top_left      = get_pixel_else_0(center,img, x-1, y-1)
				top_up        = get_pixel_else_0(center,img, x, y-1)
				top_right     = get_pixel_else_0(center,img, x+1, y-1)
				right         = get_pixel_else_0(center,img, x+1, y )
				left          = get_pixel_else_0(center,img, x-1, y )
				bottom_left   = get_pixel_else_0(center,img, x-1, y+1)
				bottom_right  = get_pixel_else_0(center,img, x+1, y+1)
				bottom_down   = get_pixel_else_0(center,img, x,   y+1 )
				mean = (top_left + top_up +top_right + right + bottom_right + bottom_down + bottom_left + left)/8
				one = top_left - bottom_right + mean
				two = top_up - bottom_down + mean
				three = top_right - bottom_left + mean
				four = right - left + mean
				five = (top_left - mean)*(bottom_right - mean)
				six = (top_up - mean)*(bottom_down - mean)
				seven = (top_right - mean)*(bottom_left - mean)
				eight = (right - mean)*(left - mean)
				values = thresholded([one + five, two + six, three + seven, four + eight])
				weights = [1, 2, 4, 8]
				res = 0
				for a in range(0, len(values)):
					res += weights[a] * values[a]

				transformed_img.itemset((x,y), res)
		plt.imshow(img,cmap = 'Greys')
		plt.show()
		plt.imshow(transformed_img,cmap = 'Greys')
		plt.show()
		hist,bins = np.histogram(transformed_img.flatten(),16,[0,16])
		LGP_features[i] = hist
def read_img_test():
	for i in range(10000):
		img = features_test[i,0:1024]
		print('processing ' + str(i))
		img = img.reshape((32,32))
		transformed_img = np.zeros((32,32))
		for x in range(0,len(img)):
			for y in range(0,len(img[0])):
				center        = img[x,y]
				top_left      = get_pixel_else_0(center,img, x-1, y-1)
				top_up        = get_pixel_else_0(center,img, x, y-1)
				top_right     = get_pixel_else_0(center,img, x+1, y-1)
				right         = get_pixel_else_0(center,img, x+1, y )
				left          = get_pixel_else_0(center,img, x-1, y )
				bottom_left   = get_pixel_else_0(center,img, x-1, y+1)
				bottom_right  = get_pixel_else_0(center,img, x+1, y+1)
				bottom_down   = get_pixel_else_0(center,img, x,   y+1 )
				mean = (top_left + top_up +top_right + right + bottom_right + bottom_down + bottom_left + left)/8
				one = top_left - bottom_right + mean
				two = top_up - bottom_down + mean
				three = top_right - bottom_left + mean
				four = right - left + mean
				five = (top_left - mean)*(bottom_right - mean)
				six = (top_up - mean)*(bottom_down - mean)
				seven = (top_right - mean)*(bottom_left - mean)
				eight = (right - mean)*(left - mean)
				values = thresholded([one + five, two + six, three + seven, four + eight])
				weights = [1, 2, 4, 8]
				res = 0
				for a in range(0, len(values)):
					res += weights[a] * values[a]

				transformed_img.itemset((x,y), res)

		hist,bins = np.histogram(transformed_img.flatten(),256,[0,256])
		LGP_features_test[i] = hist
'''
def read_img_border():
	for i in range(50000):
		img = features[i,0:1024]
		print('processing border ' + str(i))
		img = img.reshape((32,32))
		transformed_img = np.zeros((32,32))
		for x in range(0,len(img)):
			for y in range(0,len(img[0])):
				center        = img[x,y]
				top_left      = get_pixel_else_0(img, x-1, y-1)
				#top_up        = get_pixel_else_0(img, x, y-1)
				top_right     = get_pixel_else_0(img, x+1, y-1)
				#right         = get_pixel_else_0(img, x+1, y )
				#left          = get_pixel_else_0(img, x-1, y )
				bottom_left   = get_pixel_else_0(img, x-1, y+1)
				bottom_right  = get_pixel_else_0(img, x+1, y+1)
				#bottom_down   = get_pixel_else_0(img, x,   y+1 )
				values = thresholded(center, [top_left, top_right, bottom_right,
                                       bottom_left])
				weights = [1, 2, 4, 8]
				res = 0
				for a in range(0, len(values)):
					res += weights[a] * values[a]

				transformed_img.itemset((x,y), res)
		hist,bins = np.histogram(transformed_img.flatten(),16,[0,16])
		LBP_features_border[i] = hist

'''
'''
def read_img_center():
	for i in range(50000):
		img = features[i,0:1024]
		print('processing center ' + str(i))
		img = img.reshape((32,32))
		transformed_img = np.zeros((32,32))
		for x in range(0,len(img)):
			for y in range(0,len(img[0])):
				center        = img[x,y]
				#top_left      = get_pixel_else_0(img, x-1, y-1)
				top_up        = get_pixel_else_0(img, x, y-1)
				#top_right     = get_pixel_else_0(img, x+1, y-1)
				right         = get_pixel_else_0(img, x+1, y )
				left          = get_pixel_else_0(img, x-1, y )
				#bottom_left   = get_pixel_else_0(img, x-1, y+1)
				#bottom_right  = get_pixel_else_0(img, x+1, y+1)
				bottom_down   = get_pixel_else_0(img, x,   y+1 )
				values = thresholded(center, [top_up, right, left,
                                       bottom_down])
				weights = [1, 2, 4, 8]
				res = 0
				for a in range(0, len(values)):
					res += weights[a] * values[a]

				transformed_img.itemset((x,y), res)
		hist,bins = np.histogram(transformed_img.flatten(),16,[0,16])
		LBP_features_center[i] = hist
'''


#read_img_border()
#read_img_center()
read_img()
#read_img_test()
#np.savetxt("XCSLGPfeatures_train.csv",LGP_features,fmt = '%d', delimiter = ',')
#np.savetxt("XCSLGPfeatures_test.csv",LGP_features_test, fmt = '%d', delimiter = ',')
#np.savetxt("LBPcenter_features_train.csv",LBP_features_center,fmt = '%d', delimiter = ',')






