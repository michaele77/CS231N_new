#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:16:38 2020

@author: ershov
"""
#Imports
import os
import subprocess


dirList = os.listdir()
dirLen = len(dirList)
print('Directory length is: ' + str(dirLen))

userInput = input('Do you want to clone or pull from repo?')

if userInput == 'clone':
    'Cloning repo...'
    #Do all of the required pix2pix stuff
    command = ['git', 'clone', 'https://github.com/michaele77/CS231N_new']
    subprocess.run(command)
    
    # os.chdir('pytorch-CycleGAN-and-pix2pix/')
    os.chdir('CS231N-pytorch_newDir/')
    
    
    command = ['pip', 'install', '-r', 'requirements.txt']
    subprocess.run(command)
    
elif userInput == 'pull':
    
    'Pulling repo...'
    command = ['git', 'pull', 'https://github.com/michaele77/CS231N_new']
    subprocess.run(command)
    
else:
    print('Not a valid input')



#if not dirLen > 5:
#    'Cloning repo...'
#    #Do all of the required pix2pix stuff
#    command = ['git', 'clone', 'https://github.com/michaele77/CS231N_new']
#    subprocess.run(command)
#    
#    # os.chdir('pytorch-CycleGAN-and-pix2pix/')
#    os.chdir('CS231N-pytorch/')
#    
#    
#    command = ['pip', 'install', '-r', 'requirements.txt']
#    subprocess.run(command)
#    
#
#else:
#    'Pulling repo...'
#    command = ['git', 'pull', 'https://github.com/michaele77/CS231N_new']
#    subprocess.run(command)
    