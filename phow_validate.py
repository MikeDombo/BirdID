#!/usr/bin/env python

"""
Python rewrite of http://www.vlfeat.org/applications/caltech-101-code.html
This script does the same thing as phow_birdid.py (the original 
    phow_caltech101.py adapted for the BirdID project) but works with
    the methods it uses split out into a separate module.
"""
import birdid_utils
import argparse
from datetime import datetime
from os.path import exists
from scipy.io import loadmat, savemat
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm
from cPickle import dump, load

VERBOSE = True
IDENTIFIER = '20140613UR'
PREFIX = 'baseline'

# Note that the code doesn't actually check to see if the current seed is
# different from the seed that generated the existing files - basically,
# if this flag is True, all existing files will be ignored and overwritten
OVERWRITE = False  # DON'T load mat files generated with a different seed!!!

#SAMPLE_SEED = 42
#SAMPLE_SEED = 111
SAMPLE_SEED = 83150245


###############
# Main Program
###############

if __name__ == '__main__':
    ################################
    # Handle command-line arguments
    ################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_seed_arg", 
        help="Seed for choosing training sample", type=int)

    parser.add_argument("--identifier",
        help="Identifier for this data set. Should not contain the character '-'")
    
    parser.add_argument("--prefix",
        help="Tag used to distinguish versions of a data set")

    parser.add_argument("--image_dir",
                        help="Path to directory containing images")

    parser.add_argument("--num_classes",
                        help="Number of categories in image set",
                        type=int)

    parser.add_argument("--num_train",
                        help="Number of training images to use from each category",
                        type=int)

    parser.add_argument("--num_test",
                        help="Number of test images to use from each catetory",
                        type=int)

    parser.add_argument("--dsift_size",
                        action='store',
                        help="Size for vl_dsift features, follow with any number of integer values",
                        type=int,
                        nargs='*')

    parser.add_argument("--overwrite", 
                        action="store_true",
                        help="Overwrite existing model files")

    args = parser.parse_args()

    if args.sample_seed_arg:
        SAMPLE_SEED = args.sample_seed_arg
        if VERBOSE: print "SAMPLE_SEED = " + str(SAMPLE_SEED)
        
    birdid_utils.seed(SAMPLE_SEED)

    if args.identifier:
        IDENTIFIER = args.identifier
        if VERBOSE: print "IDENTIFER = " + IDENTIFIER

    if args.prefix:
        birdid_utils.REFIX = args.prefix
        if VERBOSE: print "PREFIX = " + PREFIX

    # Load default configuration
    conf = birdid_utils.Configuration(IDENTIFIER, PREFIX)

    # Update configuration from cmd line args
    if args.image_dir:
        conf.inputDir = args.image_dir
        if VERBOSE: print "Image dir: " + conf.inputDir

    if args.num_classes:
        conf.numClasses = args.num_classes
        if VERBOSE: print "numClasses = " + str(conf.numClasses)

    if args.num_train:
        conf.numTrain = args.num_train
        if VERBOSE: print "numTrain = " + str(conf.numTrain)

    if args.num_test:
        conf.numTest = args.num_test
        if VERBOSE: print "numTest = " + str(conf.numTest)

    if args.dsift_size:
        conf.phowOpts.Sizes = args.dsift_size
        if VERBOSE: print "phowOpts.Sizes = ", conf.phowOpts.Sizes
    
    if VERBOSE: print str(datetime.now()) + ' finished conf'

    if args.overwrite:
        OVERWRITE = True

    classes = birdid_utils.get_classes(conf.inputDir, conf.numClasses)
    print "Class names" , classes

    model = birdid_utils.Model(classes, conf)

    # all_images_class_labels is an array containing the integer corresponding
    # to the class the image belongs to based on the directory structure
    all_images, all_images_class_labels = birdid_utils.get_all_images(classes, conf)
    selTrain, selTest = birdid_utils.create_split(all_images, conf)
    #print "Classes " , all_images_class_labels

    if VERBOSE: print str(datetime.now()) + ' found classes and created split '


    ##################
    # Train vocabulary
    ##################
    if VERBOSE: print str(datetime.now()) + ' start training vocab'
    if (not exists(conf.vocabPath)) | OVERWRITE:
        vocab = birdid_utils.trainVocab(selTrain, all_images, conf)
        print str(datetime.now()) + ' vocab trained, saving'
        savemat(conf.vocabPath, {'vocab': vocab})
        print str(datetime.now()) + ' vocab saved'
    else:
        if VERBOSE: print 'using old vocab from ' + conf.vocabPath
        vocab = loadmat(conf.vocabPath)['vocab']
    model.vocab = vocab


    ############################
    # Compute spatial histograms
    ############################
    if VERBOSE: print str(datetime.now()) + ' start computing hists'
    if (not exists(conf.histPath)) | OVERWRITE:
        hists = birdid_utils.computeHistograms(all_images, model, conf)
        savemat(conf.histPath, {'hists': hists})
    else:
        if VERBOSE: print 'using old hists from ' + conf.histPath
        hists = loadmat(conf.histPath)['hists']


    #####################
    # Compute feature map
    #####################
    if VERBOSE: print str(datetime.now()) + ' start computing feature map'
    transformer = AdditiveChi2Sampler()
    histst = transformer.fit_transform(hists)
    train_data = histst[selTrain]
    test_data = histst[selTest]

    
    ###########
    # Train SVM
    ###########
    if (not exists(conf.modelPath)) | OVERWRITE:
        if VERBOSE: print str(datetime.now()) + ' training liblinear svm'
        if VERBOSE == 'SVM':
            verbose = True
        else:
            verbose = False
        clf = svm.LinearSVC(C=conf.svm.C)
        if VERBOSE: print clf
        clf.fit(train_data, all_images_class_labels[selTrain])
        with open(conf.modelPath, 'wb') as fp:
            dump(clf, fp)
    else:
        if VERBOSE: print 'loading old SVM model'
        with open(conf.modelPath, 'rb') as fp:
            clf = load(fp)

    ##########
    # Test SVM
    ##########   
    if (not exists(conf.resultPath)) | OVERWRITE:
        if VERBOSE: print str(datetime.now()) + ' testing svm'
        predicted_classes = clf.predict(test_data)
        true_classes = all_images_class_labels[selTest]
        accuracy = accuracy_score(true_classes, predicted_classes)
        cm = confusion_matrix(predicted_classes, true_classes)
        with open(conf.resultPath, 'wb') as fp:
            dump(conf, fp)
            dump(cm, fp)
            dump(predicted_classes, fp)
            dump(true_classes, fp)
            dump(accuracy, fp)
    else:
        with open(conf.resultPath, 'rb') as fp:
            conf = load(fp)
            cm = load(fp)
            predicted_classes = load(fp)
            true_classes = load(fp)
            accuracy = load(fp)
    

    ################
    # Output Results
    ################         
    print "accuracy =" + str(accuracy)
    print cm
    print str(datetime.now()) + ' run complete with seed = ' + str( SAMPLE_SEED )
    # Pop up a graph of the confusion matrix
    #birdid_utils.showconfusionmatrix(cm)
