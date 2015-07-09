#!/usr/bin/env python
from os.path import isdir, basename, splitext, join, isfile, getsize
from os import mkdir, remove, listdir
from glob import glob
import multiprocessing
from scipy.misc import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy import ndimage
from datetime import datetime
import argparse
import sys
from skimage import color

class Configure(object):
	def __init__(self, input_folder, output_folder, VERBOSE=False, save_figure=False, show_figure=False, threshold = 1.05, incBg = False):
		self.input_folder = input_folder
		self.output_folder = output_folder
		self.save_figure = save_figure
		self.show_figure = show_figure
		self.threshold = threshold
		self.numTot = 0
		self.incBg = incBg
		self.imPerClass = []
		self.VERBOSE = VERBOSE

def get_classes(datasetpath): #get the names of classes in the input folder
	classes_paths = [files for files in glob(datasetpath + "/*") if isdir(files)]
	classes_paths.sort()
	classes = [basename(class_path) for class_path in classes_paths]
	return classes

def get_imgfiles(path): #get the filenames in the path with the extensions listed
	all_files = []
	all_files.extend([join(path, basename(fname))
					  for fname in glob(path + "/*")
					  if splitext(fname)[-1].lower() in [".jpg", ".jpeg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]])
	return all_files

def get_all_images(classes, conf): #get a list of all the images and begin to bg remove and crop
	all_images = []
	print str(datetime.now())+" starting"
	if not isdir(conf.output_folder):
		mkdir(conf.output_folder)
	for i, imageclass in enumerate(classes):
		conf.imPerClass.append([imageclass, len(get_imgfiles(join(conf.input_folder,imageclass)))])
	for summer in conf.imPerClass:
		conf.numTot = conf.numTot+summer[1]
	for i, imageclass in enumerate(classes):
		imgs = get_imgfiles(join(conf.input_folder,imageclass))
		pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
		result = [pool.apply_async(autoCrop, args=(imName, img, imageclass, conf)) for imName, img in enumerate(imgs)]
		res = [p.get() for p in result]
		pool.terminate()
		print ""
		print str(datetime.now())+" Done "+str(imageclass)
	print str(datetime.now())+" completely done"


def trim(im, color): #crop based on the binary image to zoom into the largest area
    bg = Image.new(im.mode, im.size, 0)
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return color.crop(bbox) #actually returns the cropped image color, not im

def autoCrop(imName, img, imageclass, conf): #background remove and then crop
	imName = imName+1
	im = imread(img)
	imOrig = imread(img)
	if not isdir(conf.output_folder+"/"+imageclass):
		try:
			mkdir(conf.output_folder+"/"+imageclass)
		except:
			pass
	if imageclass == "EmptyFeeder":
		imsave(conf.output_folder+"/"+imageclass+"/"+str(imName)+"_AutoCrop_NoMod.jpg", imOrig)
		return "skipping"
	if not isfile(conf.output_folder+"/"+imageclass+"/"+str(imName)+"_AutoCrop.jpg"):
		x, y, z = im.shape
		binary_im = np.empty([x,y],np.uint8)
		r,g,b=Image.fromarray(im).getpixel((0,0))
		for i in range(0,x):
			for j in range(0,y):
				if im[i,j,1] > im[i,j,0]*conf.threshold and im[i,j,1] > im[i,j,2]*conf.threshold:
					im[i,j,:] = 255
					binary_im[i,j] = 0
				else:
					binary_im[i,j] = 1
		labels, numL = ndimage.label(binary_im) #find regions
		sizes = ndimage.sum(binary_im,labels,range(1,numL+1)) #find sizes of regions
		map = np.where(sizes==sizes.max())[0] + 1 #find largest region
		max_index = np.zeros(numL + 1, np.uint8)
		max_index[map] = 255
		max_feature = max_index[labels]
		
		if conf.incBg:
			imCrop = imOrig
		else:
			imCrop = im
		imCrop = trim(Image.fromarray(max_feature), Image.fromarray(imCrop)) #crop image
		x,y = imCrop.size #get cropped image size

		if x*y>1200000: #if cropped image is too large, then try again using HSV for bg removal
			x, y, z = imOrig.shape
			im2 = imOrig
			im2 = np.array(im2, np.float64)
			binary_im2 = np.empty([x,y], np.uint8)
			for i in range(0,x):
				for j in range(0,y):
					for k in range(0,z):
						im2[i,j,k] = im2[i,j,k]/255.0 #convert RGB from 0-255 to 0-1
			im2 = color.rgb2hsv(im2) #convert RGB to HSV
			for i in range(0,x):
				for j in range(0,y):
					for k in range(0,z):
						if k==0:
							v = 360 #change H from 0-1 to 0-360
						else:
							v = 100 #change S and V from 0-1 to 0-100
						im2[i,j,k] = int(im2[i,j,k]*v)
					if im2[i,j,0]<200 and im2[i,j,0]>60 and im2[i,j,1]>10 and im2[i,j,2]>10: #check H, S, and V if they are within the threshold
						binary_im2[i,j] = 0
					else:
						binary_im2[i,j] = 1

			labels, numL = ndimage.label(binary_im2) #find regions
			sizes = ndimage.sum(binary_im,labels,range(1,numL+1)) #find sizes of regions
			map = np.where(sizes==sizes.max())[0] + 1 #find largest region
			max_index = np.zeros(numL + 1, np.uint8)
			max_index[map] = 255
			max_feature = max_index[labels]

			if conf.incBg: #check if background should be included or removed in final output
				imCrop = imOrig
			else:
				imCrop = im
			imCrop = trim(Image.fromarray(max_feature), Image.fromarray(imCrop)) #crop image
			binary_im = binary_im2

		if conf.save_figure:
			save_figure(binary_im, labels, max_feature, imCrop, imageclass, imName, conf) #save figure
		imsave(conf.output_folder+"/"+imageclass+"/"+str(imName)+"_AutoCrop.jpg", imCrop) #save final photo

	for i, classes in enumerate(conf.imPerClass): #find the percent complete
		if classes[0] == imageclass:
			idx = i
	sys.stdout.write ("\r"+str(datetime.now())+" AutoCropped "+imageclass+" Images: "+str((imName/float(conf.imPerClass[idx][1]))*100.0)[:5]+"%") #make progress percentage
	sys.stdout.flush()
	return str(imName)

def save_figure(binary_im, labels, max_feature, imCrop, imageclass, imName, conf):
	fig = plt.figure(figsize=(15,10))
	ax = fig.add_subplot(2,2,1)
	ax2 = fig.add_subplot(2,2,2)
	ax3 = fig.add_subplot(2,2,3)
	ax4 = fig.add_subplot(2,2,4)
	ax.imshow(binary_im, cmap="gray")
	ax.set_title("Binary Image")
	ax2.imshow(labels)
	ax2.set_title("Labeled Regions")
	ax3.imshow(max_feature, cmap="gray")
	ax3.set_title("Largest Region")
	ax4.imshow(imCrop)
	ax4.set_title("Cropped Image")
	fig.set_tight_layout(True)
	if not isdir(conf.output_folder+"/figures"):
		try:
			mkdir(conf.output_folder+"/figures")
		except:
			pass
	if not isdir(conf.output_folder+"/figures/"+imageclass):
		try:
			mkdir(conf.output_folder+"/figures/"+imageclass)
		except:
			pass
	if conf.show_figure:
		plt.show()
	fig.savefig(conf.output_folder+"/figures/"+imageclass+"/figure_"+str(imName)+".png", dpi=75)
	plt.close(fig)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--threshold", help="Threshold value", type=float)
	parser.add_argument("--save_fig", help="Save Figures", type=bool)
	parser.add_argument("--show_fig", help="Show Figures", type=bool)
	parser.add_argument("--inc_bg", help="Include background in output files?", type=bool)
	parser.add_argument("--input_dir", help="Input Directory")
	parser.add_argument("--output_dir", help="Output Dataset Directory")
	
	args = parser.parse_args()
						
	if not args.input_dir or not args.output_dir:
		raise ValueError("Input or Output Dir not specified!")
				
	conf = Configure(args.input_dir, args.output_dir)
						
	if args.threshold:
		conf.threshold = args.threshold
	if args.save_fig:
		conf.save_figure = args.save_fig
	if args.show_fig:
		conf.show_figure = args.show_fig
	if args.inc_bg:
		conf.incBg = args.inc_bg


	classes = get_classes(conf.input_folder) #get the classes
	get_all_images(classes, conf) #background remove and crop the images in the classes
