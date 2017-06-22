import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
from scipy import misc
img = Image.open('police.jpg').convert('L')

img = misc.imread("police-gs",0)

print(img.shape)
import math

def sigmoid(x):
    
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = math.exp(x)
        return z / (1 + z)

def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out

def get_pixel_else_0(center,l,idx,idy,default=0):
	try:
		return l[idx,idy]
	except IndexError:
		return default

def read_img():
	for i in range(1):
		transformed_img = np.zeros((len(img),len(img[0])))
		for x in range(0,len(img)):
			for y in range(0,len(img[0])):
				center        = img[x,y]
				top_left      = int(get_pixel_else_0(center,img, x-1, y-1))
				top_up        = int(get_pixel_else_0(center,img, x, y-1))
				top_right     = int(get_pixel_else_0(center,img, x+1, y-1))
				right         = int(get_pixel_else_0(center,img, x+1, y ))
				left          = int(get_pixel_else_0(center,img, x-1, y ))
				bottom_left   = int(get_pixel_else_0(center,img, x-1, y+1))
				bottom_right  = int(get_pixel_else_0(center,img, x+1, y+1))
				bottom_down   = int(get_pixel_else_0(center,img, x,   y+1 ))
				mean = (top_left + top_up +top_right + right + bottom_right + bottom_down + bottom_left + left)/8
				one = top_left - bottom_right + center
				two = top_up - bottom_down + center
				three = top_right - bottom_left + center
				four = right - left + center
				five = (top_left - center)*(bottom_right - center)
				six = (top_up - center)*(bottom_down - center)
				seven = (top_right - center)*(bottom_left - center)
				eight = (right - center)*(left - center)
				values = thresholded(center, [top_left, top_up, top_right, right, bottom_right,
                                      bottom_down, bottom_left, left])
				weights = [1, 2, 4, 8, 16, 32, 64, 128]
				res = 0
				for a in range(0, len(values)):
					res += weights[a] * values[a]

				transformed_img.itemset((x,y), round(res))

		plt.imshow(img,cmap = 'Greys')
		plt.show()
		plt.imshow(transformed_img,cmap = 'Greys')
		plt.show()
read_img()

