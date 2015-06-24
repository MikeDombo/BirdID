#!/usr/bin/env python
from os.path import isdir, basename, splitext, join, isfile
from os import mkdir, remove, listdir
from glob import glob
import multiprocessing
from scipy.misc import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy import ndimage
from datetime import datetime

class Configure(object):
	def __init__(self, input_folder, output_folder, VERBOSE=False, save_figure=False, show_figure=False):
		self.input_folder = input_folder
		self.output_folder = output_folder
		self.save_figure = save_figure
		self.show_figure = show_figure
		self.VERBOSE = VERBOSE

def get_classes(datasetpath):
	classes_paths = [files for files in glob(datasetpath + "/*") if isdir(files)]
	classes_paths.sort()
	classes = [basename(class_path) for class_path in classes_paths]
	return classes

def get_imgfiles(path):
	all_files = []
	all_files.extend([join(path, basename(fname))
					  for fname in glob(path + "/*")
					  if splitext(fname)[-1].lower() in [".jpg", ".jpeg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]])
	return all_files

def get_all_images(classes, conf):
	all_images = []
	print str(datetime.now())+" starting"
	if not isdir(conf.output_folder):
		mkdir(conf.output_folder)
	for i, imageclass in enumerate(classes):
		imgs = get_imgfiles(join(conf.input_folder,imageclass))
		pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
		result = [pool.apply_async(remove, args=(imName, img, imageclass, conf)) for imName, img in enumerate(imgs)]
		res = [p.get() for p in result]
		print "done "+str(imageclass)
	print str(datetime.now())+" completely done"


def trim(im, color):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return color.crop(bbox)

def remove(imName, img, imageclass, conf):
	imName = imName+1
	im = imread(img)
	imOrig = imread(img)
	if not isdir(conf.output_folder+imageclass):
		try:
			mkdir(conf.output_folder+imageclass)
		except:
			pass
	if imageclass == "EmptyFeeder":
		imsave(conf.output_folder+imageclass+"/"+str(imName)+"_AutoCrop_NoMod.jpg", imOrig)
		return "skipping"
	if not isfile(conf.output_folder+imageclass+"/"+str(imName)+"_AutoCrop.jpg"):
		x, y, z = im.shape
		binary_im = np.empty([x,y],np.uint8)
		for i in range(0,x):
			for j in range(0,y):
				if im[i,j,1] > im[i,j,0]*1.05 and im[i,j,1] > im[i,j,2]*1.05:
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
		
		imCrop = trim(Image.fromarray(max_feature), Image.fromarray(imOrig))
		
		if conf.save_figure:
			save_figure(binary_im, labels, max_feature, imCrop, imageclass, imName, conf)
		imsave(conf.output_folder+imageclass+"/"+str(imName)+"_AutoCrop.jpg", imCrop)
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
	if not isdir(conf.output_folder+"figures"):
		try:
			mkdir(conf.output_folder+"figures")
		except:
			pass
	if not isdir(conf.output_folder+"figures/"+imageclass):
		try:
			mkdir(conf.output_folder+"figures/"+imageclass)
		except:
			pass
	if conf.show_figure:
		plt.show()
	fig.savefig(conf.output_folder+"figures/"+imageclass+"/figure_"+str(imName)+".png", dpi=75)
	plt.close('all')

if __name__ == "__main__":
	input_folder = "../training_2014_09_20/"
	output_folder = "../output-bg-orig/"
	conf = Configure(input_folder, output_folder, save_figure=True)
	classes = get_classes(input_folder)
	get_all_images(classes, conf)
