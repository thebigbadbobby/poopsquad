import numpy as np
from PIL import Image
import skimage
# 
# suggestion on how to structure your code... feel free to
# modify or ignore as you see fit
#

# print(True+True)
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
### Source = https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python

def image_gradient(I,sigma):
    """
    Compute the gradient orientation and magnitude after
    smoothing I with a Gaussian filter with parameter sigma
    """
#     sc
#     gaussianfilter = np.array([[ 0.002969,  0.013306,  0.021938,  0.013306,  0.002969],
#        [ 0.013306,  0.059634,  0.09832 ,  0.059634,  0.013306],
#        [ 0.021938,  0.09832 ,  0.162103,  0.09832 ,  0.021938],
#        [ 0.013306,  0.059634,  0.09832 ,  0.059634,  0.013306],
#        [ 0.002969,  0.013306,  0.021938,  0.013306,  0.002969]])
    plt.imshow(I)
    plt.show()
    gaussian_filter=matlab_style_gauss2D(shape=(5,5),sigma=sigma)
#     print(gaussianfilter)
    imsmooth = signal.convolve2d(I, gaussian_filter, mode='same')
    imdx = signal.convolve2d(imsmooth, [[-1,0,1]], mode='same') #
    imdy = signal.convolve2d(imsmooth, [[-1],[0],[1]], mode='same')
    immag = np.sqrt(imdx**2 + imdy**2);
    imdir = np.arctan(imdy/imdx)
    return immag,imdir

    

    
def detect_edge(I,sigma,thresh):
    """
    Detect edges in an image using the given smoothing and threshold
    parameters.
    """
    mag,angle = image_gradient(I,sigma)
    plt.imshow(mag,cmap='twilight')
    plt.colorbar()
    plt.show()
    plt.imshow(angle,cmap='twilight')
    plt.colorbar()
    plt.show()
    binary = mag > thresh
    return skimage.morphology.thin(binary)
#     .
#     .
#     .
image= np.array(Image.open('dilbert1.jpg'))/255
edges=detect_edge(image,5,.22)
