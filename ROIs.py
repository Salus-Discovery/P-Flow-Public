'''
Created on Mar 21, 2022
@author: jaywarrick
'''
import jbase as j
import numpy as np
import statistics
from math import sqrt
from scipy import interpolate

print('loading ROIs')
class dPoint():
	"""Represents a point with an associated integer id"""
	def __init__(self, x, y, val, n=0):
		"""Initialize point"""
		self.x = x
		self.y = y
		self.val = val
		self.n = next()
	
	def getLocation(self):
		"""Get point x and y coords"""
		return self.x, self.y, self.val, self.n

class LineProfile():
	"""Represent a line profile."""
	def __init__(self, x1, y1, x2, y2, width=None):
		"""Initialize line profile object."""
		self.x1 = x1
		self.x2 = x2
		self.y1 = y1
		self.y2 = y2
		self.width = width
    
	def getProfile(self, gray=np.array([]), step=1, length=None, method='mean', q=0.5):
		"""Get a list of x and y locations to quantify."""
		if len(gray)!=0:
			img_y = j.seq(0,len(gray)-1,step=1)
			img_x = j.seq(0,len(gray[0,:])-1,step=1)
			interp = interpolate.interp2d(img_x, img_y, gray, kind='cubic', copy=False)
		# print(str(self.x1) + str(self.y1) + str(self.x2) + str(self.y2))
		self.m = j.div(self.y2-self.y1, self.x2-self.x1)
		# Need to figure out what to do for vertical lines. Could switch to x = my+b formulation.
		self.b = self.m*(-self.x1) + self.y1
		ystep = abs(np.sin(np.arctan(-j.div(1,self.m))))
		xstep = abs(np.cos(np.arctan(-j.div(1,self.m))))
		
		if self.width > 2:
			deltaY = j.seq(0,-self.width*ystep/2,step=ystep)[::-1] + j.seq(ystep,self.width*ystep/2,step=ystep)
			deltaX = j.seq(0,-self.width*xstep/2,step=xstep)[::-1] + j.seq(xstep,self.width*xstep/2,step=xstep)
		else:
			deltaY=[0]
			deltaX=[0]
		
		if length is None:
			length = int(sqrt((self.x2-self.x1)**2 + (self.y2-self.y1)**2)+1)
			
		ret_x = j.seq(self.x1, self.x2, step=step, length=length)
		ret_y = []
		ret_z = []
		ret_n = []
		for x in ret_x:
			# Given x, find y on the profile line
			y = self.m*x+self.b
			ret_y = ret_y + [y]
			# Store the number of pixels used perpendicular to the profile line
			ret_n = ret_n + [len(deltaY)]
			i = 0
			perpz = []
			for dy in deltaY:
				perpx = x + deltaX[i]
				perpy = y + dy
				if len(gray)!=0:
					perpz = perpz + [interp(perpx, perpy)[0]]
				else:
					perpz = perpz + [0]
					i = i + 1
			# TODO: Could add wieighting function here
			if method=='mean':
				ret_z = ret_z + [statistics.mean(perpz)]
			else:
				ret_z = ret_z + [np.quantile(perpz, q)]
			
		return ret_x, ret_y, ret_z, ret_n, method
	
	def toText():
		print('updated')
