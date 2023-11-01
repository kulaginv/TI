'''
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Salt and Pepper noise.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
'''

import numpy as np

def imnoise(image,ntype="gauss",m=0,V=1,D=0.05):
    if image.ndim == 2:
        chan = 1 
        row, col = image.shape
        nsize =(row, col)
    else:
        row, col, chan=image.shape
        nsize =(row, col, chan)
    if ntype == "gauss":
        sigma = V**0.5
        noise = np.random.normal(m,sigma,nsize)
        noisy = image + noise
    elif ntype == "s&p":
        noisy = np.copy(image)
        noise = np.squeeze(np.random.rand(row,col,chan))
        print(noise.shape)
        noisy[noise<D/2] = 0
        noisy[np.logical_and(noise>=D/2,(noise<D))]= 255 if image.dtype==np.uint8 else 1
    elif ntype == "poisson":
        noisy = np.random.poisson(image)
    elif ntype == "speckle":
        noise = np.random.rand(row,col,chan)-0.5
        if image.dtype == np.uint8:
            noise=noise.astype(np.uint8)
        sigma = (12*V)**0.5
        noisy = (image+sigma*image*noise)
# trancate data
    if image.dtype == np.float64:
        noisy=np.clip(noisy,0,1)
    else:
        noisy=np.clip(noisy,0,255) 
        noisy = noisy.astype(np.uint8)   
    return noisy