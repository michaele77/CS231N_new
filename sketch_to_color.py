#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:13:41 2020

@author: ershov
"""

#Imports
import shutil 
import os
import matplotlib.pyplot as plt
import random
import subprocess


##This script takes an input sketch image, then generates a BW image using 
##the pix2pix model; uses that output as input to the colorization pix2pix model

#It can also use another model (direct) as comparison

#Initialize locations
sketch_to_bw = 'sketch_to_bw/testA/'
bw_to_color = 'bw_to_color/testA/'

#command = ['find', '.', '-name',  'datasets/combine_A_and_B.py', '
#
#find . -name '.DS_Store' -type f -delete
command = ['python', 'datasets/combine_A_and_B.py', '--fold_A', 'path_single/A', \
               '--fold_B', 'path_single/B', '--fold_AB', 'path_single']
temp = subprocess.run(command)
    
    
#Now make sure to have the latest generators saved in the checkpoints
#USe sketch_to_bw and bw_to_color as names

#PUT SKETCHES INTO TEST PHASE
#PUT THE GENERATED IMAGES THEN INTO THE VAL PHASE
command = ['python', 'test.py', '--dataroot', 'path_single', '--name', 'sketch_to_bw', \
           '--model', 'pix2pix', '--direction', 'AtoB', '--gpu_ids', 
           '-1', '--phase','test']
subprocess.run(command)
#command = ['python', 'test.py', '--dataroot', sketch_to_bw, '--name', 'sketch_to_bw', \
#          '--model', 'test', '--direction', 'AtoB', '--gpu_ids', '-1', \
#          '--netG', 'resnet_9blocks', '--norm', 'batch', '--dataset_mode', 'single']
#subprocess.run(command)


#Copy over results to the val folder, then cut out irrelevant shit
command = ['rm', '-r', 'path_single/A/val']
subprocess.run(command)
shutil.copytree('results/sketch_to_bw/test_latest/images/','path_single/A/val/')
dirList = os.listdir('path_single/A/val/')
for currFile in dirList:
    if not currFile[-10:-4] == 'fake_B':
        command = ['rm', '-r', 'path_single/A/val/' + currFile]
        subprocess.run(command)


command = ['rm', '-r', 'path_single/B/val']
subprocess.run(command)
shutil.copytree('results/sketch_to_bw/test_latest/images/','path_single/B/val/')
dirList = os.listdir('path_single/B/val/')
for currFile in dirList:
    if not currFile[-10:-4] == 'fake_B':
        command = ['rm', '-r', 'path_single/B/val/' + currFile]
        subprocess.run(command)

#recombine
command = ['python', 'datasets/combine_A_and_B.py', '--fold_A', 'path_single/A', \
               '--fold_B', 'path_single/B', '--fold_AB', 'path_single']
temp = subprocess.run(command)


command = ['python', 'test.py', '--dataroot', 'path_single', '--name', 'bw_to_color', \
           '--model', 'pix2pix', '--direction', 'AtoB', '--gpu_ids', 
           '-1', '--phase','val']
subprocess.run(command)


#Take these and store them in the output folder
command = ['rm', '-r', 'path_single/results_output']
subprocess.run(command)
shutil.copytree('results/bw_to_color/val_latest/images/','path_single/results_output/')
dirList = os.listdir('path_single/results_output/')
for currFile in dirList:
    if not currFile[-10:-4] == 'fake_B':
        command = ['rm', '-r', 'path_single/results_output/' + currFile]
        subprocess.run(command)




