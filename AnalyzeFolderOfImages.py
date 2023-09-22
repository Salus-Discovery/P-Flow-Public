# imports
import cv2
import copy
import Gen4_ImageAnalysis as lfa
import jbase as j
import math
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Open Sans']
rcParams["axes.labelweight"] = "light"
rcParams["font.weight"] = "light"
import matplotlib.pyplot as m # https://matplotlib.org/stable/tutorials/introductory/quick_start.html#sphx-glr-tutorials-introductory-quick-start-py
import pandas as pd
from zipfile import ZipFile
import datetime
import tempfile
from skimage.draw import line
import time
from pathlib import Path
import natsort
import xlsxwriter

# global variables
params = {}
imIndex = 2

def setParam(name, value):
	params[name] = value

def getParam(name, default=None):
	if name not in params.keys():
		setParam(name=name, value=default)
	return(params[name])

### Depricated parameters
setParam('ratio', float(1))			# Parameter for altering current scale with streamlit slider
setParam('oldscale', float(1))		# Temporary variable for handling changes in the ratio/scale parameter for streamlit
setParam('scale', 1.0)				# Amount to scale image before analysis (typically 1)

### Plotting and Saving parameters
setParam('folder', "G:/Shared drives/Salus Discovery/Projects/Gates LAM/Experimental Results/Gen 4 - VFA Experiments/9.14.23 Antibody screen #2/Picture") # Folder of images to analyze
setParam('resultsSubFolder', 'Python Output')	# Folder within the data folder to place results
setParam('imExtension', 'jpg')		# Only grab files with the following extension inside the specified folder
setParam('ylim',200)
setParam('date','2022.10.08')			# Date to include in results / filenames
setParam('Label','LOD')				# Label to include in results / filenames

### Analysis parameters
setParam('cropTop',152)				# Cropping dimensions of the guide window used to acquire the image
setParam('cropHeight',1800)
setParam('cropLeft',1140)
setParam('cropWidth',450)
setParam('rotate',90)				# AFTER Cropping, how much to rotate the image before profile analysis from left to right (must be a multiple of 90 and in the range of -360 to 360)
setParam('channel', 'G')				# Which channel to analyze 'R', 'G', or 'B'
setParam('stripeWidth', 110)			# Width of control and test line in [px] units
setParam('stripeSep', -600)			# Separation between control stripe and test line in [px] units
setParam('ny', '1')					# Kernel height multiplier (typical ~ 1 for Gen 4 and ~6 for Gen 3)
setParam('bg_LAMBDA', 8)				# (5-10 typical) Background subtraction radius [px]
setParam('bg_p', 6.0)				# (4-8 typical) Background sub. stiffness
setParam('presmoothFactor', 1.0)		# (0-2.0 typical) Radius of presmoothing filter
setParam('bg_win', 2)				# Background sub. width (multiplier relative to stripe width)
setParam('profileWidth', 200)			# Profile / line-scan width [px]
setParam('guessX', 0.75)				# (0-1 typical) Guess of normalized x-loc of control line
setParam('guessY', 0.5)				# (0-1 typical) Guess of normalized y-loc of control line
setParam('minSpacing', 1.5)			# (1-3 typical) Minimum spacing of peaks in terms of multiples of the stripeWidth
setParam('method', 'mean')			# ('mean' or 'quantile typical) Method used to aggregate pixel data perpendicular to line scan
setParam('quantile', 0.5)			# (0-1 typical) When using the quantile method, which quantile to select to summarize the pixels

# Get the files and initialize the results table
p = Path(getParam('folder')).glob('*.' + getParam('imExtension','jpg'))
setParam('files', [x for x in p if x.is_file()])
setParam('files', [x for _, x in natsort.natsorted(zip([x.name for x in getParam('files')], getParam('files')))]) # Sort the files by filename
setParam('data', pd.DataFrame({'Folder':[], 'File':[], 'Label':[], 'Channel':[], 'T':[], 'C':[], 'TCRatio':[]}))

# Create the output directory
Path(getParam('folder') + '/' + getParam('resultsSubFolder')).mkdir(parents=True, exist_ok=True)

# Do some error checking
if(getParam('rotate',0)%90 != 0 or getParam('rotate',0) > 360 or getParam('rotate',0) < -360):
	raise NameError('Rotation angle must be a multiple of 90 and in the range of -360 to 360')


def getNextImage():
	im = cv2.imread(str(getParam('files')[imIndex]))[:,:,channelIndex()]
	
	# Crop the image
	y = getParam('cropTop',0)
	h = getParam('cropHeight', im.shape[0]-y-1)
	x = getParam('cropLeft',0)
	w = getParam('cropWidth', im.shape[1]-x-1)
	im = im[y:(y+h), x:(x+w)]
	
	# Rotate the image
	if(getParam('rotate',0) == 90 or getParam('rotate',0) == -270):
		im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
	if(getParam('rotate',0) == -90 or getParam('rotate',0) == 270):
		im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
	if(getParam('rotate',0) == 180 or getParam('rotate',0) == -180):
		im = cv2.rotate(im, cv2.ROTATE_180)
	
	# Save the cropped image
	cv2.imwrite(str(Path(getParam('folder')) / (getParam('resultsSubFolder') + '/Crop_' + getParam('files')[imIndex].name)), im)

	# Convert to floating point for next operations
	im = im.astype(np.float16)/255.0	
	setParam('capture', im)
	updatePlots()
	
	# Aggregate results
	setParam('data', pd.concat([getParam('data'), pd.DataFrame({'Folder':[getParam('folder')], 'File':[getParam('files')[imIndex].stem], 'Label':[getParam('Label')], 'Channel':[getParam('channel')], 'T':["{:.1f}".format(getParam('test2'))], 'C':["{:.1f}".format(getParam('ctrl2'))], 'TCRatio':["{:.4f}".format(getParam('test2')/getParam('ctrl2'))], })]))
	# makeZip()

def channelIndex():
	return ['B','G','R'].index(getParam('channel','G'))

def drawLine(im, x1, y1, x2, y2, value=0):
	rr, cc = line(round((im.shape[0]-1)-y1), round(x1), round((im.shape[0]-1)-y2), round(x2))
	im[rr, cc] = value
	return

def updatePlots():
	dims = getParam('capture').shape
	
	#ProfileContent{'maxima':maxima, 'blur':blur, 'signal':prof_int, 'x1':x, 'y1':y, 'x2':gray.shape[1]-1, 'y2':y, 'width':profile_width}
	profile = lfa.getLineProfile(gray=getParam('capture'), guessX=float(getParam('guessX')), guessY=float(getParam('guessY')), minSpacing=float(getParam('minSpacing')), stripeWidth=float(getParam('stripeWidth')), ny=float(getParam('ny')), profile_width=float(getParam('profileWidth')), method=getParam('method'), q=float(getParam('quantile')), save_path=str(Path(getParam('folder')) / (getParam('resultsSubFolder') + '/StripeDetectionBlur_' + getParam('files')[imIndex].name)))
	x0 = profile['x1']
	y0 = profile['y1']
	x = np.array(profile['signal'][0])
	z = np.array(profile['signal'][2])
	daMethod = profile['signal'][4]
	
	#### FLIP THE DATA SO THE BLACK LINES BECOME WHITE LINES
	z = j.adjustIntensity(z, 0, 255, 255, 0)

	#### FIND THE "BACKGROUND" IN THE INVERTED SPACE
	# baseline params
	LAMBDA = float(getParam('stripeWidth'))*float(getParam('bg_LAMBDA'))
	p = 10.**(-1.*float(getParam('bg_p')))
	p2 = 10.**(-1.*(float(getParam('bg_p'))+1))
	presmoothSD = None if float(getParam('presmoothFactor')) < 0 else 10**float(getParam('presmoothFactor')) #float(getParam('stripeWidth'))*(10**(-1*float(getParam('presmoothFactor')))

	# quantification smooth params
	w = float(getParam('stripeWidth'))/4

	# # offsets
	# float(getParam('stripeSep')) = as.numeric(input$stripeSep)*isolate(scale$scale)
	# float(getParam('stripeWidth')) = as.numeric(input$stripeWidth)*isolate(scale$scale)

	### Calculate the baseline of the raw data ###
	z2 = j.calculateBaseline(x=z, LAMBDA=LAMBDA, p=p, niter=30, presmoothSD=presmoothSD, postsmoothSD=None)
	# st.text(str(j.adjustIntensity(z2['smooth'], oldMin=z2['bg'], oldMax=255, newMin=0, newMax=255).size))
	# st.text(str(w))
	
	### STRETCH THE DATA TO EXTEND THE BASLINE DOWN TO ZERO ###
	z2['sig_rel'] =j.adjustIntensity(z, oldMin=z2['bg'], oldMax=255, newMin=0, newMax=255)
	
	### Store a version of the data smoothed by a Gaussian for finding "peaks" of the T and C lines
	z3 = j.rollGaussian(z2['sig_rel'], w=w)

	# test.2 = max(z3[np.abs(x-(x0+float(getParam('stripeSep')))) < float(getParam('stripeWidth'))])
	# ctrl.2 = max(z3[np.abs(x-x0) < float(getParam('stripeWidth'))])
	
	### Do a localized background measurement that largely relies on the "shoulders" of the CONTROL line ###
	z4 = j.calculateBaseline(x=z2['sig_rel'][np.abs(x-x0) < float(getParam('bg_win'))*float(getParam('stripeWidth'))], LAMBDA = LAMBDA*2, p=p, niter=30, presmoothSD=presmoothSD, postsmoothSD=None)
	# st.text(str(float(getParam('bg_win'))*float(getParam('stripeWidth'))))
	# st.text(str(np.where(np.abs(x-x0) < float(getParam('bg_win'))*float(getParam('stripeWidth')))[0]))
	# st.text(str(x.size))
	z4['x'] = x[np.where(np.abs(x-x0) < float(getParam('bg_win'))*float(getParam('stripeWidth')))[0]]
	
	### Do a localized background measurement that largely relies on the "shoulders" of the TEST line ###
	z5 = j.calculateBaseline(z2['sig_rel'][np.where(np.abs(x-(x0+float(getParam('stripeSep')))) < float(getParam('bg_win'))*float(getParam('stripeWidth')))[0]], LAMBDA = LAMBDA*2, p=p, niter=30, presmoothSD = presmoothSD, postsmoothSD=None)
	z5['x'] = x[np.where(np.abs(x-(x0+float(getParam('stripeSep')))) < float(getParam('bg_win'))*float(getParam('stripeWidth')))[0]]

	# st.text(str(x))
	# st.text(str(z4['x']))
	# st.text(str(np.argmax(j.zWhereXinY(z3, x, z4['x']))))
	# st.text(str(np.argmax(j.zWhereXinY(z3, x, z4['x']))))
	ctrl2i = np.argmax(j.zWhereXinY(z3, x, z4['x']))
	test2i = np.argmax(j.zWhereXinY(z3, x, z5['x']))
	ctrl2x = int(z4['x'][ctrl2i])
	test2x = int(z5['x'][test2i])
	# st.text(f'{ctrl2i} {test2i} {ctrl2x} {test2x}')
	ctrl2hi = z3[ctrl2x]
	ctrl2lo = z4['bg'][ctrl2i]
	test2hi = z3[test2x]
	test2lo = z5['bg'][test2i]
	ctrl2 = ctrl2hi-ctrl2lo
	test2 = test2hi-test2lo
	
	# par(mar=c(6,6,1,1), mgp=c(4,1,0))
	# plot(calcs()$x, calcs()$z2$sig.rel, type='l', cex.lab=1.8, cex.axis=1.8, las=1, ylab='Signal [au]', xlab='x Location [pixels]')

	fig, ax = m.subplots()
	fig.set_figheight(4)
	fig.set_figwidth(5)
	ax.plot(x,z)
	ax.set_xlabel('Position [px]')
	ax.set_ylabel('Signal [au]')
	ax.plot(x, z2['bg'])
	m.savefig(Path(getParam('folder')) / (getParam('resultsSubFolder') + '/fig_' + getParam('files')[imIndex].stem + '.png'), dpi=300, bbox_inches='tight')
	m.close()
	setParam('fig',fig)
	
	plot, ax = m.subplots()
	plot.set_figheight(2.5)
	plot.set_figwidth(7.5)
	ax.plot(x,z2['sig_rel'])
	ax.set_xlabel('Position [px]')
	ax.set_ylabel('Signal [au]')
	ax.plot(z4['x'], z4['bg'], '-', color='orange')
	ax.plot(z5['x'], z5['bg'], '-', color='orange')
	ax.axvline(x0)
	ax.plot([ctrl2x-float(getParam('stripeWidth'))/2, ctrl2x+float(getParam('stripeWidth'))/2], [ctrl2hi,ctrl2hi], 'bo', ms=3)
	ax.plot([ctrl2x-float(getParam('stripeWidth'))/2, ctrl2x+float(getParam('stripeWidth'))/2], [ctrl2lo,ctrl2lo], 'ro', ms=3)
	ax.plot([test2x-float(getParam('stripeWidth'))/2, test2x+float(getParam('stripeWidth'))/2], [test2hi,test2hi], 'bo', ms=3)
	ax.plot([test2x-float(getParam('stripeWidth'))/2, test2x+float(getParam('stripeWidth'))/2], [test2lo,test2lo], 'ro', ms=3)
	m.ylim([0,getParam('xlim',150)])
	m.savefig(Path(getParam('folder')) / (getParam('resultsSubFolder') + '/plot_' + getParam('files')[imIndex].stem + '.png'), dpi=300, bbox_inches='tight')
	setParam('plot', plot)
	m.close()
	
	setParam('captureDraw', copy.copy(getParam('capture')))
	# Horizontal Lines
	drawLine(getParam('captureDraw'), 0, getParam('captureDraw').shape[0]-profile['y1']+profile['width']/2, profile['x2']-1, getParam('captureDraw').shape[0]-profile['y2']+profile['width']/2)
	drawLine(getParam('captureDraw'), 0, getParam('captureDraw').shape[0]-profile['y1'], profile['x2']-1, getParam('captureDraw').shape[0]-profile['y2'])
	drawLine(getParam('captureDraw'), 0, getParam('captureDraw').shape[0]-profile['y1']-profile['width']/2, profile['x2']-1, getParam('captureDraw').shape[0]-profile['y2']-profile['width']/2)
	# Vertical Lines
	drawLine(getParam('captureDraw'), profile['x1'], 0, profile['x1'], getParam('captureDraw').shape[0]-1)
	drawLine(getParam('captureDraw'), profile['x1']+float(getParam('stripeSep')), 0, profile['x1']+float(getParam('stripeSep')), getParam('captureDraw').shape[0]-1)
	setParam('captureDraw', (getParam('captureDraw')*255).astype(np.uint8))
	cv2.imwrite(str(Path(getParam('folder')) / (getParam('resultsSubFolder') + '/ImDraw_' + getParam('files')[imIndex].name)), getParam('captureDraw'))

	# # rv['results'] = rbindlist(list(isolate(rv$results), data.table(Note=isolate(as.character(input$note)), Channel=as.character(input$channel), T.peak=test.2, C.peak=ctrl.2, Ratio.peak=test.2/ctrl.2, x=x0, y=y0)))
	setParam('x0', x0)
	setParam('y0', y0)
	setParam('x', x)
	setParam('z', z)
	setParam('z2', z2)
	setParam('z3', z3)
	setParam('z4', z4)
	setParam('z5', z5)
	setParam('dims', dims)
	setParam('profile', profile)
	setParam('ctrl2x', ctrl2x)
	setParam('ctrl2hi', ctrl2hi)
	setParam('ctrl2lo', ctrl2lo)
	setParam('ctrl2', ctrl2)
	setParam('test2x', test2x)
	setParam('test2hi', test2hi)
	setParam('test2lo', test2lo)
	setParam('test2', test2)
	# return {'x0'=x0, 'y0'=y0, 'x'=x, 'z'=z, 'z2'=z2, 'z3'=z3, 'z4'=z4, 'z5'=z5, 
	#'dims'=dims, 'profile'=profile, 'ctrl2x'=ctrl2x, 'ctrl2hi'=ctrl2hi, 'ctrl2lo'=ctrl2lo, 
	#''ctrl2'=ctrl2, 'test2x'=test2x, 'test2hi'=test2hi, 'test2lo'=test2lo, 'test2'=test2,
	#''temp2'=temp2, 'stripeSep'=float(getParam('stripeSep')), 'stripeWidth'=float(getParam('stripeWidth')), 'daMethod'=daMetho}

def writeXlsx(df, sheet_name='Sheet 1'):
	path = Path(getParam('folder')) / (getParam('resultsSubFolder') + '/results.xlsx')
	writer = pd.ExcelWriter(path, engine='xlsxwriter')
	df.to_excel(writer, sheet_name=sheet_name, index=False)
	wb = writer.book
	ws = wb.get_worksheet_by_name(sheet_name)
	
	col = len(df.columns)
	ws.set_column(xlsxwriter.worksheet.xl_col_to_name(col) + ':' + xlsxwriter.worksheet.xl_col_to_name(col+1), 30)
	rows = len(df.index)
	
	for i in j.seq(0,rows-1):
		fileToGet = Path(getParam('folder')) / (getParam('resultsSubFolder') + '/ImDraw_' + df['File'].iloc[i] + getParam('files')[0].suffix)
		plotToGet = Path(getParam('folder')) / (getParam('resultsSubFolder') + '/plot_' + df['File'].iloc[i] + '.png')
		ws.insert_image(xlsxwriter.worksheet.xl_rowcol_to_cell(i+1,col),fileToGet, {'x_scale': 0.11, 'y_scale': 0.11})
		ws.insert_image(xlsxwriter.worksheet.xl_rowcol_to_cell(i+1,col+1),plotToGet, {'x_scale': 0.25, 'y_scale': 0.25})
		# ws.insert_image('A1',"C:/Users/accou/Downloads/11.8.22 LOD and U-LAM-20221115T195914Z-001/11.8.22 LOD and U-LAM/Python Output/ImDraw_0_1 x.jpg", {'x_scale': 0.1, 'y_scale': 0.1})
		ws.set_row_pixels(i+1, 55)
	wb.close()

lastIndex = len(getParam('files'))-1
# for i in seq(0,lastIndex-1):
for i in j.seq(0,lastIndex):
	imIndex = i
	getNextImage()

# Write the table of results to file
getParam('data').to_csv(str(Path(getParam('folder')) / (getParam('resultsSubFolder') + '/Results Table.csv')))
writeXlsx(getParam('data'), sheet_name='Sheet 1')

