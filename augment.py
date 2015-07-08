#!/usr/bin/env python
from os.path import isdir, basename, splitext, join, isfile
from os import mkdir, remove
from skimage import transform, util
import skimage.io as io
from scipy import ndimage
from scipy.misc import imread, imsave
from glob import glob
import multiprocessing
from PIL import Image
import random

class Configure(object):
	def __init__(self, input_folder, output_folder, VERBOSE=False):
		self.input_folder = input_folder
		self.output_folder = output_folder
		self.VERBOSE = VERBOSE

def get_classes(datasetpath):
	classes_paths = [files
					 for files in glob(datasetpath + "/*")
					 if isdir(files)]
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
	if not isdir(conf.output_folder):
		mkdir(conf.output_folder)
	for i, imageclass in enumerate(classes):
		imgs = get_imgfiles(join(conf.input_folder,imageclass))
		pool = multiprocessing.Pool(processes=4)
		result = [pool.apply_async(rotate, args=(imName, img, imageclass, conf)) for imName, img in enumerate(imgs)]
		res = [p.get() for p in result]
		print "done "+str(imageclass)
	for i, imageclass in enumerate(classes):
		imgs = get_imgfiles(join(conf.input_folder,imageclass))
		pool = multiprocessing.Pool(processes=4)
		result = [pool.apply_async(zoom, args=(imName, img, imageclass, conf)) for imName, img in enumerate(imgs)]
		res = [p.get() for p in result]
		print "done "+str(imageclass)

def rotate(imName, img, imageclass, conf):
	imName = imName+1
	i=45
	im = imread(img)
	if not isdir(conf.output_folder+imageclass):
		try:
			mkdir(conf.output_folder+imageclass)
		except:
			pass
	while i<360:
		if not isfile(conf.output_folder+imageclass+"/"+str(imName)+"_rot_"+str(i)+".jpg"):
			imsave(join(conf.output_folder,imageclass)+"/"+str(imName)+"_rot_"+str(i)+".jpg",transform.rotate(im, i, resize=False))
		elif conf.VERBOSE:
			print "skipped #"+str(imName)+" in "+str(imageclass)
		if i == 90:
			i=270
		else:
			i = i+45
	return str(imName)

def zoom(imName, img, imageclass, conf):
	imName = imName+1
	i=1.25
	im = Image.open(img)
	if not isdir(conf.output_folder+imageclass):
		try:
			mkdir(conf.output_folder+imageclass)
		except:
			pass
	if not isfile(conf.output_folder+imageclass+"/"+str(imName)+"_zoom_"+str(i)+".jpg"):
		x, y = im.size
		ims = im.crop((int((x-x/i)/2), int((y-y/i)/2), int((x+(x/i))/2), int((y+(y/i))/2)))
		imsave(join(conf.output_folder,imageclass)+"/"+str(imName)+"_zoom_"+str(i)+".jpg",ims)
	elif conf.VERBOSE:
		print "skipped #"+str(imName)+" in "+str(imageclass)
	return str(imName)

if __name__ == "__main__":
	input_folder = "../output-nobg-1.05/"
	output_folder = "../augmented/"
	conf = Configure(input_folder, output_folder, VERBOSE=False)
	classes = get_classes(input_folder)
	get_all_images(classes, conf)