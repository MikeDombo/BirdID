# BirdID
==================

Script for content based image classification using the bag of visual words approach, based on the phow_caltech101.py port by Ludwig Schmidt-Hackenberg, which was itself based on the phow_caltech101.m Matlab script by Andrea Vedaldi.

The script is a Python version of [phow_caltech101.m][1], a 'one file' example script using the [VLFeat library][6] to train and evaluate an image classifier 
on the [Caltech-101 data set][4]. It has been adapted to use a set of images of birds rather than the Caltech 101 image library.

Like the original Matlab version this Python script achieves the same (State-of-the-Art in 2008) average accuracy of around 65% as the original file:

- PHOW features (dense multi-scale SIFT descriptors)
- Elkan k-means for fast visual word dictionary construction
- Spatial histograms as image descriptors
- A homogeneous kernel map to transform a Chi2 support vector machine (SVM) into a linear one
- Liblinear SVM (instead of the Pegasos SVM of the Matlab script)

The code also works with other datasets if the images are organized like in the Calltech data set, where all images belonging to one class are in the same folder:
    
    .
    |-- path_to_folders_with_images
    |    |-- class1
    |    |    |-- some_image1.jpg
    |    |    |-- some_image1.jpg
    |    |    |-- some_image1.jpg
    |    |    └ ...
    |    |-- class2
    |    |    └ ...
    |    |-- class3
        ...
    |    └-- classN

There are no constraints for the names of the files or folders. File extensions can be configured in [`conf.extensions`][7] But note that the code fails with a segmentation fault (on Mac OS X 10.8.5, at least) when the images are PNGs.

Changes from phow_caltech101.py:

- Added command line argument --sample_seed_arg to set the seed used for generating the random split of training and test images
- Added command line argument --identifier to set the data set
identification tag
- Added the --prefix command line argument for the prefix to distinguish between
versions of a data set
- Added the --image_dir command line argument to allow the path to the folder
containing the images to use to be specified at runtime
- Added the --num_classes command line argument to allow the number of classes
represented in the data set to be specified at runtime
- Added the --num_train command line argument to allow the number of training
images to use from each class to be specified at runtime
- Added the --num_test command line argument to allow the number of test
examples to use from each class to be specified at runtime
- Added the --dsift_size command line argument to allow the number of features
used in the vl_dsift method to be specified at runtime.

Also added two additional files to the project:

phow_train.py

This file does only the training stage of the process, writing out the 
constructed model for use in classification or validation. Uses the
birdid_utils module, which contains helper code extracted from 
phow_birdid.py

phow_validate.py:

This file does the same thing as phow_birdid.py, but it works with the 
birdid_utils module rather than being self-contained.

Requisite:

- [VLFeat with a Python wrapper][2]
- [scikit-learn][5] to replace VLFeat ML functions that don't have a Python wrapper yet. 
- [The Caltech101 dataset][3]

[5]: http://scikit-learn.org/stable/
[4]: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
[2]: https://pypi.python.org/pypi/pyvlfeat/
[3]: http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
[1]: http://www.vlfeat.org/applications/caltech-101-code.html
[6]: http://www.vlfeat.org/index.html
[7]: https://github.com/shackenberg/phow_caltech101.py/blob/master/phow_caltech101.py#L58
