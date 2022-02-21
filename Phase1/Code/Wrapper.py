#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park
"""

# Code starts here:
import numpy as np
import cv2
import matplotlib.pyplot as plt 
from math import sqrt
from sklearn.cluster import KMeans
import time
import scipy as sp 
from scipy  import signal 
from glob import glob

def showFilterBank(filter_bank, row, col, filename) :
	fig, ax = plt.subplots(row, col)
	filt_num = 0
	for r in range(row) : 
		for c in range(col) :
			ax[r, c].axis('off')
			ax[r, c].imshow(filter_bank[filt_num], cmap='gray')
			filt_num += 1

	fig.tight_layout()
	plt.axis('off')
	plt.savefig(f"{filename}.jpg")
	plt.close('all')


def Convolve2d(image, kernel) : 
	img_h, img_w = image.shape[0], image.shape[1]
	kernel_h, kernel_w = kernel.shape 

	output_image = np.zeros(image.shape)

	pad_h = int((kernel_h)/2)
	pad_w = int((kernel_w)/2)

	pad_img = np.zeros((img_h+(2*pad_h), img_w+(2*pad_w)))
	pad_img[pad_h:pad_img.shape[0]-pad_h, pad_w:pad_img.shape[1]-pad_w] = image

	for row in range(img_h) : 
		for col in range(img_w) : 
			output_image[row, col] = np.sum(kernel * pad_img[row:row+kernel_h, col:col+kernel_w])

	return output_image

def Gaussian2d(sigma, size) :
	sigmax, sigmay = sigma, sigma 
	x, y = np.meshgrid(np.linspace(-(size//2), (size//2), size), np.linspace(-(size//2), (size//2), size))
	term1 = (1/sigmax)*(np.exp(-np.square(x)/(2*np.square(sigmax))))
	term2 = (1/sigmay)*(np.exp(-np.square(y)/(2*np.square(sigmay))))

	final = (1/(2*np.pi)*(term1*term2))
	return final 

def Gaussian2dDer1(sigma, size) : 
	sigmax, sigmay = sigma, 3*sigma
	x, y = np.meshgrid(np.linspace(-(size//2), (size//2), size), np.linspace(-(size//2), (size//2), size))
	gx = (1/sigmax)*(np.exp(-np.square(x)/(2*np.square(sigmax))))
	gy = (1/sigmay)*(np.exp(-np.square(y)/(2*np.square(sigmay))))

	# gxx = d(gx)*gy
	gxx = (-x/np.square(sigmax))*gx
	final = gxx*gy

	return final

def Gaussian2dDer2(sigma, size) :
	sigmax, sigmay = sigma, 3*sigma
	x, y = np.meshgrid(np.linspace(-(size//2), (size//2), size), np.linspace(-(size//2), (size//2), size))
	gx = (1/sigmax)*(np.exp(-np.square(x)/(2*np.square(sigmax))))
	gy = (1/sigmay)*(np.exp(-np.square(y)/(2*np.square(sigmay))))

	# gxxx = d(d(gx))*gy = d(gxx)*gy
	gxxx = ((np.square(x)/sigmax**4)-(1/sigmax**2))*gx 
	final = gxxx*gy 
	
	return final 

def DoG(orientations, scales) :
	Sx = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
	Sy = np.array([[1, 2, 1],[0, 0, 0], [-1, -2, -1]])

	filter_bank = []
	
	for scale in range(1,scales+1) : 
		scales = scale
		gaussian = Gaussian2d(scales, 30)

		Gx = cv2.filter2D(gaussian, -1, Sx)
		Gy = cv2.filter2D(gaussian, -1, Sy)

		for orientation in range(orientations) : 
			angle = (2 * np.pi * orientation/orientations)
			G = Gx*np.cos(angle) + Gy*np.sin(angle)
			filter_bank.append(G)
			
	return filter_bank

def Gaborfun(Lambda, Psi, Gamma, Sigma, Angle) : 
	lambdas, psi, gamma = Lambda, Psi, Gamma
	sigma_x = Sigma 
	sigma_y = sigma_x/gamma
	angle = Angle

	size = 30
	x, y = np.meshgrid(np.linspace(-(size//2), (size//2), size), np.linspace(-(size//2), (size//2), size))

	x_theta = x*np.cos(angle)+y*np.sin(angle)
	y_theta = -x*np.sin(angle)+y*np.cos(angle)

	s_term = np.cos((2*np.pi*x_theta/lambdas)+psi)
	f_term = Gaussian2d(Sigma, size)

	gb = f_term*s_term

	return gb 

def Gabor(scales, orientations) :
	filter_bank = []
	Lambda, Psi, Gamma = 10, 0, 1

	for scale in range(5, scales+5, 2) : 
		Sigma = scale
		gb = Gaborfun(Lambda, Psi, Gamma, Sigma, np.pi/2)

		for orientation in range(orientations) : 
			Angle = 360*(orientation/orientations)
			rows, cols = gb.shape
			M = cv2.getRotationMatrix2D((cols/2, rows/2), Angle, 1)
			gabor = cv2.warpAffine(gb, M, (cols, rows))
			filter_bank.append(gabor)
	
	return filter_bank

def LM(scales) : 
	orientations = 6 
	fder, sder, filter_bank, der, gaus, LoG = [], [], [], [], [], []
	derscales = scales[0:3]

	for scale in derscales : 
		gaussian1 = Gaussian2dDer1(scale, 30)
		gaussian2 = Gaussian2dDer2(scale, 30)

		for orientation in range(orientations) :
			angle = orientation*(180/orientations)

			rows, cols = gaussian1.shape
			M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
			rgaussian1 = cv2.warpAffine(gaussian1, M, (cols, rows))
			der.append(rgaussian1)

		for orientation in range(orientations) :
			angle = orientation*(180/orientations)
			rows, cols = gaussian2.shape
			M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
			rgaussian2 = cv2.warpAffine(gaussian2, M, (cols, rows))
			der.append(rgaussian2)

	for scale in scales : 
		gaussian = Gaussian2d(scale, 20)
		gaus.append(gaussian)

	lap_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
	up_scales = [3*i for i in scales]
	log_scales = scales + up_scales
	for scale in log_scales : 
		gaussian = Gaussian2d(scale, 30)
		log = cv2.filter2D(gaussian, -1, lap_kernel)
		LoG.append(log)

	filter_bank = der + gaus + LoG
	
	return filter_bank


def half_discs(orientations, scales) : 
	half_discs, left_discs, right_discs  = [], [], []
	for scale in range(2,2+scales) : 
		radius = 4 * scale
		kernel = np.zeros((2*radius+1, 2*radius+1))
		y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
		mask = x**2 + y**2 <= radius**2
		kernel[mask] = 1 

		rows, cols = kernel.shape
		for r in range(rows//2, rows) :
			for c in range(0,cols) :
				kernel[r][c] = 0

		for orientation in range(orientations) :
			angle1 = orientation*(360/orientations)
			angle2 = angle1+180

			M1 = cv2.getRotationMatrix2D((cols//2, rows//2), angle1, 1)
			rotate1 = cv2.warpAffine(kernel, M1, (cols, rows))
			half_discs.append(rotate1)
			left_discs.append(rotate1)

			M2 = cv2.getRotationMatrix2D((cols//2, rows//2), angle2, 1)
			rotate2 = cv2.warpAffine(kernel, M2, (cols, rows))
			half_discs.append(rotate2)
			right_discs.append(rotate2)
	
	return half_discs, left_discs, right_discs 

def createPixelColorIds(imagePath, K) : 
	image = cv2.imread(imagePath) 
	h, w, _ = image.shape 
	image = image.reshape(h*w, 3)

	clusterspixels = KMeans(n_clusters=K, random_state=0).fit(image).labels_
	clusterspixels = clusterspixels.reshape((h, w))

	return clusterspixels 

def createPixelTextonIds(imagePath, K, filter_banks) : 
	image = cv2.imread(imagePath, 0)
	h, w = image.shape
	filternum = len(filter_banks)

	filter_imgs = np.zeros((filternum, h, w))
	for k, filter_bank in enumerate(filter_banks) : 
		fimage = cv2.filter2D(image, -1, filter_bank)
		filter_imgs[k] = fimage

	filter_imgs = filter_imgs.reshape((filternum, h*w))
	clusterspixels = KMeans(n_clusters=K, random_state=0).fit_predict(np.transpose(filter_imgs))
	clusterspixels = clusterspixels.reshape((h,w))

	return clusterspixels

def createPixelBrightIds(imagePath, K) : 
	image = cv2.imread(imagePath, 0) 
	h, w = image.shape 
	image = image.reshape(h*w, 1)

	clusterspixels = KMeans(n_clusters=K, random_state=0).fit(image).labels_
	clusterspixels = clusterspixels.reshape((h, w))

	return clusterspixels 

def printoutput(img, name):
	plt.figure()
	plt.imshow(img, cmap="hsv")
	plt.axis('off')
	plt.savefig(f"{name}.jpg")

def ComputeChiSqr(g_i, h_i) : 
	return 0.5*((g_i-h_i)**2)/(g_i+h_i+np.exp(-5))

def generateGradient(Map, left_discs, right_discs, K) : 
	n, m = Map.shape 
	p = len(right_discs)
	t_g = np.zeros((n, m, p))

	num = 0 
	for left, right in list(zip(left_discs, right_discs)) :
		h, w = Map.shape
		img = np.zeros((h, w))
		chi_sqr = np.zeros((h, w))
		for Bin in range(K) : 
			val = np.where(Map==Bin)
			img[val] = 1 
			# img[Map==Bin] = 1
			
			g_i = cv2.filter2D(img, -1, left)
			h_i = cv2.filter2D(img, -1, right)

			chi_sqr += ComputeChiSqr(g_i, h_i)

		t_g[:, :, num] = chi_sqr
		num += 1
	return t_g
	
def main():
	main_folder = "./"

	canny_path = main_folder + "BSDS500/CannyBaseline"
	sobel_path = main_folder + "BSDS500/SobelBaseline"
	imgs_path = main_folder + "BSDS500/Images"

	output_filters = main_folder + "Output/Filters/"
	output_gradients = main_folder + "Output/Gradients/"
	output_maps = main_folder + "Output/Maps/"
	output_pblite =  main_folder + "Output/PbLite/"


	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	row, col = 2, 16
	filter_bankDoG = DoG(col, row) 
	showFilterBank(filter_bankDoG, row, col, f"{output_filters}" + "DoG")

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""

	row, col = 4, 12
	scaleLMS = [1, sqrt(2), 2, 2*sqrt(2)]
	scaleLML = [sqrt(2), 2, 2*sqrt(2), 4]

	filter_bankLMS = LM(scaleLMS) 
	# filter_bankLML = LM(scaleLML) 
	showFilterBank(filter_bankLMS, row, col, f"{output_filters}" + "LMS")
	# showFilterBank(filter_bankLML, row, col, "LML")

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	scales, orientations = 8, 6
	filter_bankGabor = Gabor(scales, orientations)
	showFilterBank(filter_bankGabor, 4, orientations, f"{output_filters}" + "Gabor")

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	scales, orientations = 3, 8
	halfdiscs, left_discs, right_discs = half_discs(orientations, scales)
	showFilterBank(halfdiscs, scales*2, orientations, f"{output_filters}" + "Half-Discs")

	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""
	filter_banks = filter_bankLMS + filter_bankDoG + filter_bankGabor

	imgs_list = glob(imgs_path + "/*.jpg")
	c_list = glob(canny_path + "/*.png")
	s_list = glob(sobel_path + "/*.png")

	for i in range(len(imgs_list)) :
		KT = 64
		imagePath = imgs_list[i]
		TextonMap = createPixelTextonIds(imagePath, KT, filter_banks)
		printoutput(TextonMap, f'{output_maps}' + f'Texture{i+1}')

		"""
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""

		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		grT = generateGradient(TextonMap, left_discs, right_discs, KT)
		grT = np.mean(grT, axis = 2)
		fig = plt.figure()
		plt.imshow(grT)
		plt.axis('off')
		plt.savefig(f"{output_gradients}" + f"Gradient-Texton{i+1}.jpg")
		plt.close('all')


		"""
		Generate Brightness Map
		Perform brightness binning 
		"""

		KB = 16
		BrightnessMap = createPixelBrightIds(imagePath, KB)
		printoutput(BrightnessMap, f'{output_maps}' + f'Brightness{i+1}')

		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		grB = generateGradient(BrightnessMap, left_discs, right_discs, KB)
		grB = np.mean(grB, axis = 2)
		fig = plt.figure()
		plt.imshow(grB)
		plt.axis('off')
		plt.savefig(f"{output_gradients}" + f"Gradient-Brightness{i+1}.jpg")
		plt.close('all')

		"""
		Generate Color Map
		Perform color binning or clustering
		"""
		KC = 16
		ColorMap = createPixelColorIds(imagePath, KC)
		printoutput(ColorMap, f'{output_maps}' + f'Color{i+1}')
		
		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""

		grC = generateGradient(ColorMap, left_discs, right_discs, KC)
		grC = np.mean(grC, axis = 2)
		fig = plt.figure()
		plt.imshow(grC)
		plt.axis('off')
		plt.savefig(f"{output_gradients}" + f"Gradient-Color{i+1}.jpg")
		plt.close('all')

		h, w = grC.shape[0], grC.shape[1]
		"""
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""
		sobelPb = cv2.imread(s_list[i], 0)

		"""
		Read Canny Baseline
		use command "cv2.imread(...)"
		"""
		cannyPb = cv2.imread(c_list[i], 0)

		"""
		Combine responses to get pb-lite output
		Display PbLite and save image as PbLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""
		a = (grB+grC+grT)/3
		pb_edge = (a)*(0.2*sobelPb+0.8*cannyPb)
		# cv2.imwrite(f'Pb_Edge{i}.jpg', pb_edge)

		fig = plt.figure()
		plt.imshow(pb_edge, cmap='gray')
		plt.axis('off')
		plt.savefig(f'{output_pblite}' + f'PbEdge{i+1}.jpg')
		plt.close('all')

if __name__ == '__main__':
	main()
 


