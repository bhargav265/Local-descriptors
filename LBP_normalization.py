import numpy as np 
import pandas as pd 

LBP_features_train = pd.read_csv("LBP_features_train.csv",header = None)
LBP_features_test =  pd.read_csv("LBP_features_test.csv",header = None)
LGP_features_train =  pd.read_csv("LGPfeatures_train.csv",header = None)
LGP_features_test = pd.read_csv("LGPfeatures_test.csv",header = None)

LBP_features_train = LBP_features_train.values.astype(np.float)

LBP_features_test = LBP_features_test.values.astype(np.float)
LGP_features_train = LGP_features_train.values.astype(np.float)
LGP_features_test = LGP_features_test.values.astype(np.float)

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
#abels = np.concatenate((labels1,labels2,labels3,labels4,labels5),axis = 0)

#labels1 = labels1.reshape(10000,1)
#print(labels1.shape)
#print(features.shape)


#print("Features " + str(features.shape))
#print("Labels "+ str(labels1.shape))
#features = features.astype(np.int)
#print(features[0])
#print(features.shape)
#LBP_features = np.zeros((10000,256))



def normalise(arr):
	a = arr
	#print("Max = " + str(max(a)) + " Min = "+ str(min(a)))
	for i in range(len(a)):
		a[i] = (arr[i] - min(arr))/(max(arr)-min(arr))
		#print((arr[i] - min(arr))/(max(arr)-min(arr)))
		#print(' ')
	return a
def transform():
	for i in range(LBP_features_train.shape[0]):
		print("normalizing training set " + str(i))
		LBP_features_train[i] = normalise(LBP_features_train[i])
		#print(LBP_features_train[i])
	
	for i in range(LBP_features_test.shape[0]):
		print("normalizing test set " + str(i))
		LBP_features_test[i] =normalise(LBP_features_test[i])
	for i in range(LGP_features_train.shape[0]):
		print("normalizing training set " + str(i))
		LGP_features_train[i] = normalise(LGP_features_train[i])
	for i in range(LGP_features_test.shape[0]):
		print("normalizing test set " + str(i))
		LGP_features_test[i] = normalise(LGP_features_test[i])
transform()



'''
def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out

def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx,idy]
    except IndexError:
        return default

def read_img():
	for i in range(10000):
		img = features[i,0:1024]
		print('processing ' + str(i))
		img = img.reshape((32,32))
		transformed_img = np.zeros((32,32))
		for x in range(0,len(img)):
			for y in range(0,len(img[0])):
				center        = img[x,y]
				top_left      = get_pixel_else_0(img, x-1, y-1)
				top_up        = get_pixel_else_0(img, x, y-1)
				top_right     = get_pixel_else_0(img, x+1, y-1)
				right         = get_pixel_else_0(img, x+1, y )
				left          = get_pixel_else_0(img, x-1, y )
				bottom_left   = get_pixel_else_0(img, x-1, y+1)
				bottom_right  = get_pixel_else_0(img, x+1, y+1)
				bottom_down   = get_pixel_else_0(img, x,   y+1 )
				values = thresholded(center, [top_left, top_up, top_right, right, bottom_right,
                                      bottom_down, bottom_left, left])
				weights = [1, 2, 4, 8, 16, 32, 64, 128]
				res = 0
				for a in range(0, len(values)):
					res += weights[a] * values[a]

				transformed_img.itemset((x,y), res)
		hist,bins = np.histogram(transformed_img.flatten(),256,[0,256])
		LBP_features[i] = hist
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



read_img_border()
read_img_center()
'''

np.savetxt("LBP_features_train_normal.csv",LBP_features_train, delimiter = ',')
np.savetxt("LBP_features_test_normal.csv",LBP_features_test, delimiter = ',')
np.savetxt("LGP_features_train_normal.csv",LGP_features_train, delimiter = ',')
np.savetxt("LGP_features_test_normal.csv",LGP_features_test, delimiter = ',')






