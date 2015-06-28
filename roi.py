#!/usr/bin/env python
from os.path import isdir, basename, splitext, join, isfile
from os import mkdir, remove, listdir
from glob import glob
import multiprocessing
from PIL import Image
from skimage.morphology import disk
from skimage.filters.rank import mean, median
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
from datetime import datetime

class Configure(object):
	def __init__(self, input_folder, output_folder, VERBOSE=False):
		self.input_folder = input_folder
		self.output_folder = output_folder
		self.VERBOSE = VERBOSE
		self.mean_threshold = 40

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
	print str(datetime.now())+" starting"
	if not isdir(conf.output_folder):
		mkdir(conf.output_folder)
	for i, imageclass in enumerate(classes):
		imgs = get_imgfiles(join(conf.input_folder,imageclass))
		pool = multiprocessing.Pool(processes=4)
		result = [pool.apply_async(zoom, args=(imName, img, imageclass, conf)) for imName, img in enumerate(imgs)]
		res = [p.get() for p in result]
		print "done "+str(imageclass)
	print str(datetime.now())+" completely done"


def zoom(imName, img, imageclass, conf):
	imName = imName+1
	im = Image.open(img)
	if not isdir(conf.output_folder+imageclass):
		try:
			mkdir(conf.output_folder+imageclass)
		except:
			pass
	fileMatch = 0
	for file in listdir(conf.output_folder+"/"+imageclass):
		if fnmatch.fnmatch(file, str(imName)+"_zoom_"+'*.jpg'):
			fileMatch = fileMatch+1
			if fileMatch>=3:
				if conf.VERBOSE: print "exists, so breaking: #"+str(imName)+" in "+str(imageclass)
				return str(imName)
	x, y = im.size
	x1=0
	y1=0
	means=[]
	while y1<=y-480:
		while x1<=x-640:
			ims = im.crop((x1, y1, x1+640, y1+480))
			mean1 = mean(np.array(ims)[:,:,1], disk(700))
			means.append(((x1,y1),int(mean1[0][0])))
			x1 = x1+160
		x1=0
		y1 = y1+120
	zoomNRC(imName, img, imageclass, conf, conf.mean_threshold, means, im)
	return str(imName)

def zoomNRC(imName, img, imageclass, conf, threshold, means, im):
	x, y = im.size
	x1=0
	y1=0
	numBird = 0
	while y1<=y-480:
		while x1<=x-640:
			ims = im.crop((x1, y1, x1+640, y1+480))
			idx = 0
			for i, a in enumerate(means):
				if a[0]==(x1,y1):
					idx=i
			if (idx>0):
				if (abs(means[idx][1]-means[idx-1][1])>threshold):
					if (x1==0 and y1>0):
						id = 0
						for i, a in enumerate(means):
							if a[0]==(x1,y1-120):
								id=i
						if (abs(means[idx][1]-means[id][1])>threshold):
							ims.save(join(conf.output_folder,imageclass)+"/"+str(imName)+"_zoom_"+str(x1)+"_"+str(y1)+".jpg")
							numBird = numBird+1
						else:
							x1 = x1+160
							continue
					elif (x1==0 and y1==0):
						id = 0
						for i, a in enumerate(means):
							if a[0]==(x1,y1+120):
								id=i
						if (abs(means[idx][1]-means[id][1])>threshold):
							ims.save(join(conf.output_folder,imageclass)+"/"+str(imName)+"_zoom_"+str(x1)+"_"+str(y1)+".jpg")
							numBird = numBird+1
					else:
						ims.save(join(conf.output_folder,imageclass)+"/"+str(imName)+"_zoom_"+str(x1)+"_"+str(y1)+".jpg")
						numBird = numBird+1
			x1 = x1+160
		x1=0
		y1 = y1+120
	if(numBird<3 and threshold>0):
		if conf.VERBOSE:
			print "trying harder with "+str(threshold-1)+" on image #"+str(imName)+" in category "+str(imageclass)
		zoomNRC(imName, img, imageclass, conf, threshold-1, means, im)
	else:
		print "done: #"+str(imName)+" in "+str(imageclass)+" using threshold of "+str(threshold)
	return str(imName)

if __name__ == "__main__":
	input_folder = "../training_2014_09_20/"
	output_folder = "../output-G/"
	conf = Configure(input_folder, output_folder, VERBOSE=False)
	classes = get_classes(input_folder)
	get_all_images(classes, conf)