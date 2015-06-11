#!/usr/bin/env python
from os.path import isdir, basename, splitext, join
from os import mkdir, remove
from scipy import ndimage
from scipy.misc import imread, imsave, imresize
from glob import glob
import multiprocessing


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

def get_all_images(classes, input_folder, output_folder):
	all_images = []
	if not isdir(output_folder):
		mkdir(output_folder)
	for i, imageclass in enumerate(classes):
		imgs = get_imgfiles(join(input_folder,imageclass))
		pool = multiprocessing.Pool(processes=4)
		result = [pool.apply_async(augment, args=(imName, output_folder, img, imageclass)) for imName, img in enumerate(imgs)]
		res = [p.get() for p in result]
		print "done "+str(imageclass)

def augment(imName, output_folder, img, imageclass):
	imName = imName+1
	i=0
	im = imread(img)
	if im.shape[1]>480:
		im = imresize(im, (640, 480))
	if not isdir(output_folder+imageclass):
		mkdir(output_folder+imageclass)
	while i<360:
		imsave(join(output_folder,imageclass)+"/"+str(imName)+"_rot_"+str(i)+".jpg",ndimage.rotate(im, i))
		i = i+45
	return "done "+str(imName)

if __name__ == "__main__":
	input_folder = "training_2014_09_20/"
	output_folder = "output/"
	classes = get_classes(input_folder)
	get_all_images(classes, input_folder, output_folder)