"""
CLAB Task-1: Harris Corner Detector
Your name (Your uniID): u6734495
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imread
from skimage.color import rgb2gray
import cv2


def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    return result


def fspecial(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Parameters, add more if needed
sigma = 2
thresh = 0.07

# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
dy = dx.transpose()


bw = imread('Harris_1.jpg')
bw=rgb2gray(bw)
bw= (bw * 255).astype(int)
# computer x and y derivatives of image
Ix = conv2(bw, dx)
Iy = conv2(bw, dy)

g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)

Iy2 = conv2(np.power(Iy, 2), g)
Ix2 = conv2(np.power(Ix, 2), g)
Ixy = conv2(Ix * Iy, g)

######################################################################
# Task: Compute the Harris Cornerness
######################################################################

img_1=np.array(Image.open('Harris_1.jpg'))
img_2 = np.array(Image.open('Harris_2.jpg'))
img_3=np.array(Image.open('Harris_3.jpg'))
img_4=np.array(Image.open('Harris_4.jpg'))


def cornerness(img,window_size):
#     det= Ix2 * Iy2 - (Ixy**2)
#     tr= Ix2 + Iy2

#     R = det - thresh * (tr ** 2)
    R = np.zeros((img.shape[0]-window_size + 1,img.shape[1]-window_size +1))
    height = R.shape[0]
    width = R.shape[1]
    for y in range(height):
        for x in range(width):
            Sxx = np.sum(Ix2[y:y + window_size , x : x+window_size])
            Syy = np.sum(Iy2[y:y + window_size , x : x+window_size])
            Sxy = np.sum(Ixy[y:y + window_size , x : x+window_size])
            det= Sxx * Syy - (Sxy**2)
            tr= Sxx + Syy
            response = det - thresh * (tr ** 2)
            R[y,x] = response
    return R


######################################################################
# Task: Perform non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
######################################################################


def nonMaximalSupress(image,kernel_size):
    M, N= image.shape
    local=[]
    coord=[]
    for x in range(0,M-2):
        for y in range(0,N-2):
            window = image[x:x+kernel_size, y:y+kernel_size]
            localMax = np.amax(window)
            
            if localMax >R[R>0].mean():
                maxCoord = list(np.unravel_index(np.argmax(window, axis=None), window.shape))
                maxCoord[0]+=x
                maxCoord[1]+=y
                
                coord.append(maxCoord)
                local.append(localMax)
    coord_matrix = np.reshape(coord,(len(coord),2))
                
            
    return coord_matrix

R = cornerness(img_1,3)
corner_coords= nonMaximalSupress(R,3)

im= np.array(Image.open('Harris_1.jpg'))
# radius = 1
# thickness = -2
# color = (255,0,0)
# for i in corner_coords:
#     image = cv2.circle(im,(i[1],i[0]), radius, color, thickness)
# plt.imshow(image)
for i in corner_coords:
    im[i[0],i[1]]=[255,0,0]
plt.imshow(im)
#cy =Image.fromarray(im)
#cy.save("Corner_4.jpg")

#Creating inbuilt Harris function to test our function against

image = cv2.imread('Harris_1.jpg') 
operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
operatedImage = np.float32(operatedImage) 
dest = cv2.cornerHarris(operatedImage, 2,3, 0.07) 
dest = cv2.dilate(dest, None) 
image[dest > 0.01 * dest.max()]=[255, 0,0] 
#Image.fromarray(image).save("Cv2 corner_1.jpg")



# Dispalying all the saved files that we got from running our
#and inbilt functions


output_1=np.array(Image.open('Corner_1.jpg'))
output_3=np.array(Image.open('Corner_3.jpg'))
output_4=np.array(Image.open('Corner_4.jpg'))
output_2=np.array(Image.open('Corner_2.jpg'))
fig,axes = plt.subplots(1,2 , figsize = (20,10))
inbuilt_1 =np.array(Image.open('Cv2 corner_1.jpg'))

axes[0].imshow(output_1)
axes[0].set_title("My Harris",fontsize = 25)
axes[1].imshow(inbuilt_1)
axes[1].set_title("Inbuilt Harris",fontsize =25)

fig,axes = plt.subplots(1,2 , figsize = (20,10))
inbuilt_3 =np.array(Image.open('Cv2 corner_3.jpg'))

axes[0].imshow(output_3)
axes[0].set_title("My Harris",fontsize = 25)
axes[1].imshow(inbuilt_3)
axes[1].set_title("Inbuilt Harris",fontsize = 25)

fig,axes = plt.subplots(1,2 , figsize = (20,10))
inbuilt_4 =np.array(Image.open('Cv2 corner_2.jpg'))

axes[0].imshow(output_4)
axes[0].set_title("My Harris",fontsize = 25)
axes[1].imshow(inbuilt_4)
axes[1].set_title("Inbuilt Harris",fontsize =25)

fig,axes = plt.subplots(1,2 , figsize = (20,10))
inbuilt_2 =np.array(Image.open('Cv2 corner_4.jpg'))

axes[0].imshow(output_2)
axes[0].set_title("My Harris",fontsize = 25)
axes[1].imshow(inbuilt_2)
axes[1].set_title("Inbuilt Harris",fontsize =25)


plt.show()
