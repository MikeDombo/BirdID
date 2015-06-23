#!/usr/bin/env python
from os.path import isdir, basename, splitext, join, isfile
from os import mkdir, remove
from scipy.misc import imread, imsave, imresize
from glob import glob
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class Configure(object):
	def __init__(self, input_folder, output_folder, source, VERBOSE=False):
		self.input_folder = input_folder
		self.output_folder = output_folder
		self.src = source
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
	if not isdir(conf.output_folder):
		mkdir(conf.output_folder)
	for i, imageclass in enumerate(classes):
		imgs = get_imgfiles(join(conf.input_folder,imageclass))
		for ii, im in enumerate(imgs):
			ii = ii+1
			if not isdir(conf.output_folder+imageclass):
				try:
					mkdir(conf.output_folder+imageclass)
				except:
					pass
			if not isfile(conf.output_folder+imageclass+"/"+str(ii)+"histo"+".jpg"):
				s = imread(im)
				h = histogram_adjust(s, imread(conf.src))
				imsave(conf.output_folder+imageclass+"/"+str(ii)+"histo"+".jpg", h(s))
			#"""
				hist1, bin1 = np.histogram(s, bins=256)
				hist2, bin2 = np.histogram(imread(conf.output_folder+imageclass+"/"+str(ii)+"histo"+".jpg"), bins=256)
				hist3, bin3 = np.histogram(imread(conf.src), bins=256)
				center1 = (bin3[:-1] + bin3[1:]) / 2
				fig = plt.figure(figsize=(15,10))
				ax = fig.add_subplot(2,2,1)
				ax2 = fig.add_subplot(2,2,2)
				ax3 = fig.add_subplot(2,2,3)
				ax4 = fig.add_subplot(2,2,4)
				ax3.bar(center1, hist1, align='center', color='red')
				ax3.bar(center1, hist3, align='center', color='green')
				ax3.set_title("Input and Aim")
				ax4.bar(center1, hist2, align='center', color='red')
				ax4.bar(center1, hist3, align='center', color='green')
				ax4.set_title("Output and Aim")
				ax.bar(center1, hist1, align='center', color='red')
				ax.set_title("Input")
				ax2.bar(center1, hist2, align='center', color='green')
				ax2.set_title("Output")
				#plt.show()
				if not isdir(output_folder+"figures"):
					try:
						mkdir(conf.output_folder+"figures")
					except:
						pass
				if not isdir(output_folder+"figures/"+imageclass):
					try:
						mkdir(output_folder+"figures/"+imageclass)
					except:
						pass
				fig.savefig(output_folder+"figures/"+imageclass+"/figure_"+str(ii)+".png", dpi=600)
				plt.close('all')
			#"""
		print "done "+str(imageclass)


def histogram_adjust(source, target):
	mask = (source > 0) & (target > 0)
	
	s_hist, s_edges = np.histogram(source[mask], bins=256)
	t_hist, t_edges = np.histogram(target[mask], bins=256)
	
	s_edges = s_edges[:-1]
	t_edges = t_edges[:-1]
	
	s_cum = np.cumsum(s_hist)
	t_cum = np.cumsum(t_hist)
	
	mapping = np.argmin(np.abs(t_cum - s_cum[:, None]), axis=1)
	
	def tf(source):
		source = source.astype(float)
		sshape = source.shape
		source = source.flat
		source = source/np.max(source) * (len(s_edges) - 1)
		source = np.clip(np.round(source), 0, len(s_edges) - 1).astype(int)
		return t_edges[mapping[source]].reshape(sshape) + \
			(t_edges[1] - t_edges[0])/2.
	return tf

if __name__ == "__main__":
	input_folder = "../training_2014_09_20/"
	output_folder = "../test-histo/"
	source = input_folder+"EmptyFeeder/2014-03-06_12.31.03-1.jpg"
	conf = Configure(input_folder, output_folder, source, VERBOSE=False)
	classes = get_classes(input_folder)
	get_all_images(classes, conf)
