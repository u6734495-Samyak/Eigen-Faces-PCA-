import numpy as np
import random
import math
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, color
import cv2

im1=Image.open('mandm.png')
lab1= color.rgb2lab(im1)
im2 =Image.open('peppers.png')
lab2 = color.rgb2lab(im2)

def get_centroids(im,k):
    """input : Takes an image im and value k for number of clusters
                to be made
       output : outputs randomly chosen k centroids in the image.
    """   
    centers=[]
    for i in range(k):
        centers.append(random.choice(random.choice(im)))
    return centers

def min_dist(pix,centroids):
    """ input : Takes a pixel value and the  k centroids as input
        output : returns the index of the centroid from which the minimum
                distance to the pixel was closest.
    """
    dist=[]
    for i in centroids:
        dist.append(np.sum(np.power((i-pix),2)))
    min_dist=np.argmin(dist)
    return min_dist

def find_clusters(im,centroids):
    """ Input : Takes the image and the k centroids as inputs
        output:  returns a dictionary in which keys are clusters numbers and values 
                are the pixels associated to that cluster.
    """
    keys=[]
    for x in range(len(centroids)):
        keys.append(x)
    clusters ={key:[] for key in keys} 

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            p = im[i,j]
            min_index = min_dist(p,centroids)
            clusters[min_index].append(p)
            
    return clusters

def new_cent(centroids,clusters):
    """input : takes the centroids and clusters as inputs
       output : finds the new centre by calcualting the mean of all pixels 
               in the associated cluster.
    """
    newcent=[]
    key=clusters.keys()
    for i in key:
        n_mean= np.mean(clusters[i],axis=0)
        newcent.append(n_mean)
    return newcent

def kpp(im1,K):
    cent=[]
    x=random.randrange(im1.shape[0])
    y=random.randrange(im1.shape[1])
    cent.append(im1[x,y])
    pixels =[]
    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            p =[]
            p.append(list(im1[i,j]))
            pixels.append(p)
    d=[]
    for k in range(1,K):
        dist=[]
        for x in range(im1.shape[0]):
            for y in range(im1.shape[1]):
                for c in cent:
                    d.append(np.sum((im1[x,y]-c)**2))
                dist.append(min(d))
                d=[]
        prob=dist/np.sum(dist)
        cum_prob=np.cumsum(prob)
        random_sample=random.random()
        for j,l in enumerate(cum_prob):
            if random_sample < l:
                i=pixels[j]
                break
        cent.append(i)
    return cent

def k_means(im,k):
    """input : takes an image and the number of clusters needed
       output : an image segmented into the k clusters
    
    """
    ima=np.array(im)
    h=ima.shape[0]
    w=ima.shape[1]
    #Ucomment the next line for runniing the random intialization
    #centroids= get_centroids(ima,k)
    #Next line is to get the outputs from Kmeans++ initilaization
    centroids= kpp(ima,k)

    old_centroids=[]
    for i in range(20):
        print(i)
        old_centroids = centroids 
        clusters = find_clusters(ima,centroids)
        centroids = new_cent(old_centroids,clusters)
    for x in range(h):
        for y in range(w):
            ima[x,y]=centroids[min_dist(ima[x,y],centroids)]
    fig = plt.figure()
    fig.suptitle(" K =15 for kmeans ++  on RGB image")
    plt.imshow(ima)
k_means(im1,15)

def make_lab5d(im):
    lab5d = np.zeros((im.shape[0],im.shape[1],5))
    height = lab5d.shape[0]
    width =lab5d.shape[1] 
    for x in range(height):
        for y in range(width):
            lab5d[x,y,0:3]=im[x,y,0:3]
            lab5d[x,y,3]=x
            lab5d[x,y,4]=y
    return lab5d

def k_means_for_5d(im,k):
    ima = make_lab5d(im)
    h=ima.shape[0]
    w=ima.shape[1]
    #Uncomment the next line for runniing the random intialization
    #centroids= get_centroids(ima,k)
    #Next line is to get the outputs from Kmeans++ initilaization
    centroids = kpp(ima,k)
    old_centroids=[]
    for i in range(20):
        print(i)
        old_centroids = centroids
        clusters = find_clusters(ima,centroids)
        centroids = new_cent(old_centroids,clusters)
    for x in range(h):
        for y in range(w):
            ima[x,y]=centroids[min_dist(ima[x,y],centroids)]
    fig = plt.figure()
    fig.suptitle(" K =15 for kmeans ++  on LAB 5d image")
    plt.imshow(color.lab2rgb(ima[:,:,0:3]))
    # pic = Image.fromarray((color.lab2rgb(ima[:,:,0:3])*255).astype('uint8'))
    # pic.save(" mandm k =10 with c kmeans++(lab5d).png")
k_means_for_5d(lab1,15)

plt.show()






