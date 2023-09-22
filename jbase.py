'''
Created on Mar 19, 2022

@author: jaywarrick
'''
import math
import numpy as np
import cv2 as cv
import skimage
import ipdb
from datatable import dt, f, update, by, sort
from skimage.feature import peak_local_max
from scipy import sparse
from scipy.sparse.linalg import spsolve
import copy
import pandas as pd
import threading 
import time
# from streamlit.scriptrunner.script_run_context import add_script_run_ctx

left_weights = None
left_threshold = None
right_weights = None
right_threshold = None

def resizeIm(x, scale, anti_aliasing=True, preserve_range=False):
	return skimage.transform.rescale(image=x, scale=scale, anti_aliasing=anti_aliasing, preserve_range=preserve_range)

def seq(start, stop, step=1, length=None, precision=10):
    """Create a vector of numbers"""
    
    def isGood(item, start, stop, precision=10):
        return ((item-stop) <= 0 and (item-start) >= 0 ) or ((item-stop) >= 0 and (item-start) <= 0 ) or round(item-stop, precision) == 0
    
    if length!=None:
        ret = np.linspace(start, stop, length)
        return ret.tolist()
    else:
        length = round(1+abs((stop-start)/step))
        if stop < start and step > 0:
            step = -1*step
        return list(filter(lambda item: isGood(item=item, start=start, stop=stop, precision=precision), np.arange(0,length)*step+start))

def div(a, b, precision=10):
        """Safe division"""
        # import math
        if abs(b-0) < 1**-precision:
            return math.inf
        else:
            return a/b

def getWeight(x, x50, p=2):
	return(1/(1+pow(x/(x50), p)))

def getControlXY(gray, radMax=350, p=2, delta=0.0000001):
    """Find the XY location of the control stripe"""
    h = len(gray)
    w = len(gray[0])
    guessRows=h/2
    guessCols=w/3
    leftBias = gray.astype(np.float32)/255.
    temp = gray.astype(np.float32)/255.
    daMax = np.max(gray)
    daMin = np.min(gray)
    for i in seq(0,len(gray)-1):
        for j in seq(0,len(gray[0])-1):
            if(gray[i][j]==0):
                temp[i][j]=delta
            else:
                blah1 = 1/(1+pow(1-(daMax-gray[i][j])/(daMax-daMin), p))
                r = math.sqrt((guessRows-i)*(guessRows-i)+(guessCols-j)*(guessCols-j))
                blah2 = 1/(1+pow((r)/(radMax), p))
                temp[i][j] = blah1*blah2
    #             # print("[" + str(i) + "," + str(j) + "] = " + str(blah))
    mult = 1.2
    portion = temp[int(guessRows-mult*radMax):int(guessRows+mult*radMax),int(guessCols-mult*radMax):int(guessCols+mult*radMax)]
    th2 = cv.threshold((portion*255.).astype('uint8'),0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    print(th2[0])
    
    # Get the position of the spot.
    x = range(0, th2[1].shape[0])
    y = range(0, th2[1].shape[1])
    (X,Y) = np.meshgrid(x,y)
    denom = th2[1].sum().astype("float")
    x_coord = (X*th2[1]).sum() / denom
    y_coord = (Y*th2[1]).sum() / denom
    th2,temp2 = cv.threshold((temp*255).astype('uint8'),int(th2[0]),255,cv.THRESH_BINARY)
    # print(str(temp))
    retRow = guessRows-mult*radMax+y_coord
    retCol = guessCols-mult*radMax+x_coord
    print(retRow)
    print(retCol)
    return (temp*255).astype('uint8'), temp2, th2, retRow, retCol

def getMaxima(gray, minSpacing, invert=False, closestToRC=None):
	"""Get the maxima of an image and return a datatable of their locations and intensity in decreasing order of intensity"""
	daRows=[]
	daCols=[]
	value=[]
	if(invert==True):
		if(np.max(gray) <= 1):
			blah = peak_local_max(255-gray*255, minSpacing)
		else:
			blah = peak_local_max(255-gray, minSpacing)
	else:
		blah = peak_local_max(gray, minSpacing)
	if(blah.shape[0] == 0):
		return(dt.Frame(dict(r=[], c=[], value=[])))
	for i in seq(0,max(blah.shape[0]-1,0)):
		daRows.append(blah[i][0])
		daCols.append(blah[i][1])
		value.append(float(gray[daRows[-1],daCols[-1]]))
	ret = dt.Frame(dict(r=daRows, c=daCols, value=value))
	if(closestToRC is None):
		if(invert==True):
			ret = ret[:,:,sort(f.value)]
		else:
			ret = ret[:,:,sort(-f.value)]
	else:
		ret[:, update(dist=dt.math.sqrt(dt.math.pow(f.r-closestToRC[0],2)+dt.math.pow(f.c-closestToRC[1],2)))]
		ret = ret[:,:,sort(f.dist)]
	ret[:, 'id']=np.array(seq(1,ret.nrows))
	return ret

def rank_simple(vector):
    return sorted(range(len(vector)), key=vector.__getitem__)

def getRanks(a):
    n = len(a)
    ivec=rank_simple(a)
    svec=[a[rank] for rank in ivec]
    sumranks = 0
    dupcount = 0
    newarray = [0]*n
    for i in range(n):
        sumranks += i
        dupcount += 1
        if i==n-1 or svec[i] != svec[i+1]:
            averank = sumranks / float(dupcount) + 1
            for j in range(i-dupcount+1,i+1):
                newarray[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newarray

# Convert to something like
def rollGaussian(x, w):
	if x is None:
		return None
	x = np.array(x)
	# Sigma is considered w/2
	x2 = np.insert(x, 0, np.repeat(x[0], 5*round(w)))
	x2 = np.append(x2, np.repeat(x[-1], 5*round(w)))
	gx = np.array(seq(-5*round(w), 5*round(w)))
	gaussian = np.exp(-(gx/(w/2))**2/2)
	gaussian = np.divide(gaussian, np.sum(gaussian))
	result = np.convolve(x2, gaussian, mode="same")
	return result[5*round(w):(5*round(w)+x.size)]

# import numpy as np
# from scipy import sparse
# from scipy.sparse.linalg import spsolve
def baselineALS(x, LAMBDA, p, niter=10):
    L = len(x)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = LAMBDA * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*x)
        w = p * (x > z) + (1-p) * (x < z)
    return {'baseline':z, 'corrected':np.subtract(x, z), 'w':w}



def calculateBaseline(x, LAMBDA=3, p=0.002, niter=30, presmoothSD=None, postsmoothSD=None):
	if presmoothSD is not None:
		x = rollGaussian(x=x, w=presmoothSD)
	
	ret = baselineALS(
		x=x,
		LAMBDA=LAMBDA,
		p=p,
		niter = niter
	)
	
	sig = ret['corrected']
	bg = ret['baseline']
	
	if postsmoothSD is not None:
		sig = rollGaussian(x=sig, w=postsmoothSD)
		bg = x-sig

	return pd.DataFrame({'sig':sig, 'bg':bg, 'wgts':ret['w'], 'smooth':x})

def _adjustIntensity(x, oldMin, oldMax, newMin, newMax):
	ratio = (newMax - newMin) / (oldMax - oldMin)
	offset = newMin - ratio*oldMin
	ret = x * ratio + offset
	return(ret)

def adjustIntensity(x, oldMin, oldMax, newMin, newMax):
	temp = np.vectorize(_adjustIntensity)
	return temp(x, oldMin, oldMax, newMin, newMax)

def whereXinY(x, y):
	if not isnp(x):
		x = np.array(x)
	if not isnp(y):
		y = np.array(y)
	return(np.where(np.isin(x, y))[0])

def xiny(x, y):
	if not isnp(x):
		x = np.array(x)
	if not isnp(y):
		y = np.array(y)
	return whereXinY(x,y)

def zWhereXinY(z, x, y):
	if not isnp(x):
		x = np.array(x)
	if not isnp(y):
		y = np.array(y)
	if not isnp(z):
		z = np.array(z)
	return z[whereXinY(x,y)]

def isnp(x):
	type(x).__module__ == np.__name__

# class RepeatedTimer(object):
# 	def __init__(self, interval, duration, function, includeStreamlitContext, *args, **kwargs):
# 		self._timer = None
# 		self.interval = interval
# 		self.function = function
# 		self.args = args
# 		self.kwargs = kwargs
# 		self.is_running = False
# 		self.start_time = None
# 		self.next_time = None
# 		self.end_time = None
# 		self.duration = duration
# 		self.includeStreamlitContext = includeStreamlitContext
# 		# self.start()
# 	
# 	def _run(self):
# 		self.is_running = False
# 		self.start()
# 		self.function(*self.args, **self.kwargs)
# 		# hello('World', self.end_time-self.start_time, self.next_call-self.start_time)
# 	
# 	def start(self):
# 		if not self.is_running:
# 			if self.start_time is None:
# 				self.start_time = time.time()
# 				self.next_time = self.start_time
# 				self.end_time = self.start_time + self.duration
# 			if time.time() <= (self.end_time+0.1): # Add 0.1 s for some wiggle room to be sure to capture the last frame
# 				self.next_time += self.interval
# 				self._timer = threading.Timer(self.next_time - time.time(), self._run)
# 				if self.includeStreamlitContext:
# 					add_script_run_ctx(self._timer)
# 				self._timer.start()
# 				self.is_running = True
# 	
# 	def stop(self):
# 		self.start_time = None
# 		self.end_time = None
# 		self._timer.cancel()
# 		self.is_running = False

# def hello(name, total=5, itr=1):
# 	print('Hello {} - {} - {}!'.format(str(name), str(total), str(itr)))
# import jbase as j
# rt = j.RepeatedTimer(1,5,print,False,'myWords')
