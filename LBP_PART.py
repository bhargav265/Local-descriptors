import struct
import numpy as np
import matplotlib.pyplot as plt
#import cv2



def read_image(file_name, idx_image):
	
	img_file = open(file_name,'r+b')
	#print(img_file)
	
	img_file.seek(0)
	magic_number = img_file.read(4)
	magic_number = struct.unpack('>i',magic_number)
	#print('Magic Number: '+str(magic_number[0]))
		
	data_type = img_file.read(4)
	data_type = struct.unpack('>i',data_type)
	#print('Number of Images: '+str(data_type[0]))


	dim = img_file.read(8)
	dimr = struct.unpack('>i',dim[0:4])
	dimr = dimr[0]
	#print('Number of Rows: '+str(dimr))
	dimc = struct.unpack('>i',dim[4:])
	dimc = dimc[0]
	#print('Number of Columns:'+str(dimc))


	image = np.ndarray(shape=(dimr,dimc))
	img_file.seek(16+dimc*dimr*idx_image)
	
	for row in range(dimr):
		for col in range(dimc):
			tmp_d = img_file.read(1)
			tmp_d = struct.unpack('>B',tmp_d)
			image[row,col] = tmp_d[0]
	
	img_file.close()
	return image



def thresholded(center,pixels):
	out = []
	for a in pixels:
		if a >= center:
			out.append(1)
		else:
			out.append(0)
	
	return out

def get_pixel_else_0(l,idx,idy,default = 0):
	try:
		return l[idx,idy]
	except IndexError:
		return default

def LBP_border(dataset,i):
	img = read_image(dataset,i)
	timg = read_image(dataset,i)

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
				timg.itemset((x,y), res)
	hist,bins = np.histogram(timg.flatten(),256,[0,256])
	return hist
def LBP_center(dataset,i):
	img = read_image(dataset,i)
	timg = read_image(dataset,i)

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
			values = thresholded(center, [top_up, right, bottom_down,
            	                        left])
			weights = [1, 2, 4, 8]
			res = 0
			for a in range(0, len(values)):
				res += weights[a] * values[a]
				timg.itemset((x,y), res)
	hist,bins = np.histogram(timg.flatten(),256,[0,256])
	return hist
#plt.hist(timg.flatten(),256,[0,256], color = 'r')

def LBP_extract_features(dataset,length):
	features = np.zeros((length,256))
	for i in range(0,length):
		res = LBP(dataset,i)
		features[i] = res
		if(i%1000==0):
			print('finished extracting features for %d image'% (i))
	return features

x_test = LBP_extract_features('t10k-images.idx3-ubyte',10000)

print(x_test.shape)

np.savetxt('LBP_features_test.csv',x_test,delimiter = ",",fmt ='%d')
#print(x_train)
