# import sys
import math
import numpy as np
import cv2 as cv
# from matplotlib import pyplot as plt
from datatable import dt, f, update, by
import jbase as j
import ROIs
import skimage.io
import skimage.color
import skimage.filters
import skimage.measure
from skimage.transform import rescale

def getStripeKernel(stripeWidth=125, ny=5):
	stripeWidth = int(stripeWidth)
	stripeHeight = int(stripeWidth*ny)
	if(stripeWidth % 2 == 0):
		stripeWidth = stripeWidth + 1
	if(stripeHeight % 2 == 0):
		stripeHeight = stripeHeight + 1
	k = np.ones((stripeHeight, stripeWidth*3), dtype=np.float64)
	k[:,j.seq(stripeWidth,stripeWidth*2-1)] = 0
	k = k/(np.sum(k))
	k[:,j.seq(stripeWidth,stripeWidth*2-1)] = -1/((stripeWidth)*(stripeHeight))
	return k,stripeWidth,stripeHeight

def blurStripe(gray, stripeWidth=125, scale=1.0, ny=5, invert=False):
	
	k,tempWidth,tempHeight = getStripeKernel(stripeWidth=stripeWidth*scale, ny=ny)
	if(invert==True):
		ret = cv.filter2D(src=1-gray, ddepth=-1, kernel=k)
	else:
		ret = cv.filter2D(src=gray, ddepth=-1, kernel=k)
	
	# set the top and bottom border to 0 to ensure the peaks are not on the edge
	# of the image
	blackWidth = int(0.2*stripeWidth)
	ret[0:blackWidth,:] = 0
	ret[(ret.shape[0]-blackWidth-1):(ret.shape[0]-1),:] = 0
	ret = cv.GaussianBlur(ret,(tempWidth,tempHeight*2+1),0)
	return ret

def getMaxNearXY(gray, guessX, guessY, minSpacing, invert=False):
	# Get local maxima nearest the guess location
	maxima = j.getMaxima(gray, minSpacing=int(minSpacing), invert=invert, closestToRC=(gray.shape[0]*guessY, gray.shape[1]*guessX))
	
	if maxima.shape[0]==0:
		return -1, -1, maxima
	
	# sort "cleverly" to get the peack we want.
	maxima.cbind(dt.Frame({'DistRank':j.getRanks(maxima['dist'].to_numpy())}))
	maxima['newDist'] = maxima[:, f.value / f.DistRank]
	maxima = maxima[:, :, dt.sort('newDist', reverse=True)]
	
	# grab the highest ranking peak
	x = maxima[0,'c']
	y = maxima[0,'r']
	
	
	# Threshold the image to half the max of the closes maxima
	th3,ret3 = cv.threshold(gray,maxima[0,'value']/2,255,cv.THRESH_BINARY)
	
	# Label the thresholded regions
	labeled_image, count = skimage.measure.label(ret3.astype(bool), connectivity=2, return_num=True)
	props = dt.Frame(skimage.measure.regionprops_table(labeled_image, properties=('label','area','centroid')))
	
	# Find the label corresponding to the closest maxima
	labelToGet = labeled_image[y,x]
	
	# Return the centroid of that thresholded region
	y = np.array(props[f.label==labelToGet,:]['centroid-0'])[0,0]
	x = np.array(props[f.label==labelToGet,:]['centroid-1'])[0,0]
	return x, y, maxima

def getLineProfile(gray, guessX=0.333, guessY=0.5, minSpacing=0.1, stripeWidth=125, ny=1, profile_width=125, method='mean', q=0.5, save_path=None):
	"""Return a tuple that is 1) the thresholded and labeled image as numpy array, 2) a line profile measurement as a numpy vector, and 3) the dictionary of line profile parameters (x1, y1, x2, y2, width)"""
	
	# img = cv.imread(img_path)
	if np.max(gray) <= 1:
		gray = (gray*255).astype('uint8')
	else:
		gray = gray.astype('uint8') #cv.cvtColor(np.asarray(img), cv.COLOR_BGR2GRAY)
	
	# blur = cv.GaussianBlur(gray,(25,25),0)
	# th3,ret3 = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
	# th3 = th3*thresh_mult
	# th3,ret3 = cv.threshold(blur,th3,255,cv.THRESH_BINARY_INV)
	# # perform connected component analysis
	# labeled_image, count = skimage.measure.label(ret3.astype(bool), connectivity=2, return_num=True)
	# # plt.imshow(gray)
	# # plt.title('gray')
	# 
	# props = dt.Frame(skimage.measure.regionprops_table(labeled_image, properties=('label','area','centroid')))
	# props.names = {'centroid-0':'row', 'centroid-1':'col'}
	# props[:, update(dist=dt.math.sqrt(dt.math.pow(f.row-gray.shape[0]*guessY, 2) + dt.math.pow(f.col-gray.shape[1]*guessX, 2))), by('label')]
	# props[:, update(score=f.area/(f.dist**loc_weight))]
	# x = props[f.score==dt.max(f.score),'col'].to_list()[0][0]
	# y = props[f.score==dt.max(f.score),'row'].to_list()[0][0]
	# print('here')
	# print(gray)
	blur = blurStripe(gray=gray, stripeWidth=stripeWidth, scale=1, ny=ny, invert=False)
	if save_path is not None:
		cv.imwrite(save_path, j.adjustIntensity(blur, 0, np.max(blur), 0, 255))
	
	# print(blur)
	# print('now here')
	x,y,maxima = getMaxNearXY(blur, guessX=guessX, guessY=guessY, minSpacing=stripeWidth*minSpacing, invert=False)
	# x,y,maxima = getControlXY(gray=gray, guessX=guessX, guessY=guessY, stripeWidth=stripeWidth, invert=True)
	
	if maxima.shape[0]==0:
		return {'maxima':maxima, 'blur':blur, 'signal':None, 'x1':x, 'y1':y, 'x2':gray.shape[1]-1, 'y2':y, 'width':profile_width}
	
	prof = ROIs.LineProfile(0, y, gray.shape[1]-1,y, width=profile_width)
	prof_int = prof.getProfile(gray=gray, method=method, q=q)
	
	return {'maxima':maxima, 'blur':blur, 'signal':prof_int, 'x1':x, 'y1':y, 'x2':gray.shape[1]-1, 'y2':y, 'width':profile_width}
	# plt.imshow(gray)
	# plt.axhline(y=y-width/2,color='red', lw=1, linestyle='--')
	# plt.axhline(y=y,color='red', lw=2)
	# plt.axhline(y=y+width/2,color='red', lw=1, linestyle='--')
	# plt.axvline(x=x,color='red', lw=1, linestyle='--')
	# plt.axvline(x=x+r.stripeSep,color='red', lw=1, linestyle='--')
	# plt.title('gray')

