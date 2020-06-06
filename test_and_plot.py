#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:28:38 2020

@author: ershov
"""


#We want to flesh out 3 aspects during testing:
#-Inception Scores for validation and testing images
#-Image creation scheme (tiles 3-5 images, sketch + BW + gen, for some model)
#-being able to choose between different models (not just latest one)
#-plotting losses


#Imports
import shutil 
import os
import matplotlib.pyplot as plt
import subprocess
from distutils.dir_util import copy_tree
import cv2
import numpy as np


#command = ['python', 'datasets/combine_A_and_B.py', '--fold_A', 'path/to/data/A', \
#           '--fold_B', 'path/to/data/B', '--fold_AB', 'path/to/data']
#subprocess.run(command)






#Define loading and plotting functions

def lossDataLoad(inStr):

  G_GAN = []
  G_loss = []
  D_real = []
  D_fake = []

  with open(inStr + '/loss.txt', 'r') as filehandle:
      #It's a list of lists of floats
      fullFile = filehandle.readlines()
  cntr = 0
  for i in fullFile:
    cntr += 1
    if i == '-\n':
      cntr = 0
      continue
    elif cntr == 1:
      G_GAN.append(float(i[:-2]))
    elif cntr == 2:
      G_loss.append(float(i[:-2]))
    elif cntr == 3:
      D_real.append(float(i[:-2]))
    elif cntr == 4:
      D_fake.append(float(i[:-2]))

  epoch = []
  epoch_iter = [] 
  with open(inStr + '/epoch.txt', 'r') as filehandle:
      #It's a list of 2 numbers for epoch
      fullFile = filehandle.readlines()
  cntr = 0
  for i in fullFile:
    if i == '-\n':
      cntr = 0
      continue
    else:
      cntr += 1
      if cntr == 1:
        epoch.append(float(i[:-2]))
      else:
        epoch_iter.append(float(i[:-2]))
  combEpoch = []
  for i in range(len(epoch)):
    bigE = epoch[i]
    endE = epoch_iter[-1]

    temp = (bigE - 1)*endE + epoch_iter[i]
    combEpoch.append(temp)

  return G_GAN, G_loss, D_real, D_fake, combEpoch


#define running average helper
def running_mean(inList, N):
    x = np.array(inList)
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)



#plotting function
def lossPlotFunc(G_GAN, G_loss, D_real, D_fake, combEpoch, path, name):
  runWindow = 10
  run_G_L1 = running_mean(G_GAN, runWindow)
  run_G_loss = running_mean(G_loss, runWindow)
  run_D_real = running_mean(D_real, runWindow)
  run_D_fake = running_mean(D_fake, runWindow)
    
  run_D_delta = run_D_real - run_D_fake
  run_G_obj = run_G_L1 + run_G_loss
  combEpoch = combEpoch[:-9]
  
  
#  run_G_obj = list(run_G_obj)
#  run_G_L1 = list(run_G_L1)
#  run_G_loss = list(run_G_loss)
#  run_D_real = list(run_D_real)
#  run_D_fake = list(run_D_fake)
#  run_D_delta = list(run_D_delta)
  

  fig, axs = plt.subplots(1, 2, figsize=(15,5))
  axs[0].plot(combEpoch, run_G_L1)
  axs[0].set_title(name + ': Generator L1 Error', fontsize = 20)
  axs[0].set_xlabel('Iterations', fontsize = 20)
  axs[0].set_ylabel('Error', fontsize = 20)
  axs[0].grid(True)

  axs[1].plot(combEpoch, run_G_loss)
  axs[1].set_title(name + ': Generator Loss', fontsize = 20)
  axs[1].set_xlabel('Iterations', fontsize = 20)
  axs[1].set_ylabel('Loss', fontsize = 20)
  axs[1].grid(True)
  plt.savefig(path + name + 'loss_G.jpg')

  fig, axs = plt.subplots(1, 2, figsize=(15,5))
  axs[0].plot(combEpoch, run_D_delta)
  axs[0].set_title(name + ': Discr Critic Loss', fontsize = 20)
  axs[0].set_xlabel('Iterations', fontsize = 20)
  axs[0].set_ylabel('Confidence', fontsize = 20)
  axs[0].grid(True)

  #This one is the actual objective function!
  axs[1].plot(combEpoch, run_G_obj)
  axs[1].set_title(name + ': Total GAN Objective', fontsize = 20)
  axs[1].set_xlabel('Iterations', fontsize = 20)
  axs[1].set_ylabel('Confidence', fontsize = 20)
  axs[1].grid(True)
  plt.savefig(path + name + 'loss_D.jpg')

#  fig, axs = plt.subplots(1, 2, figsize=(15,5))
#  axs[0].plot(combEpoch, G_GAN)
#  axs[0].set_title(name + ': Generator Error', fontsize = 20)
#  axs[0].set_xlabel('Iterations', fontsize = 20)
#  axs[0].set_ylabel('Error', fontsize = 20)
#  axs[0].grid(True)
#
#  axs[1].plot(combEpoch, G_loss)
#  axs[1].set_title(name + ': Generator Loss', fontsize = 20)
#  axs[1].set_xlabel('Iterations', fontsize = 20)
#  axs[1].set_ylabel('Loss', fontsize = 20)
#  axs[1].grid(True)
#  plt.savefig(path + name + 'loss_G.jpg')
#
#  fig, axs = plt.subplots(1, 2, figsize=(15,5))
#  axs[0].plot(combEpoch, D_real)
#  axs[0].set_title(name + ': Discr Real Confidence', fontsize = 20)
#  axs[0].set_xlabel('Iterations', fontsize = 20)
#  axs[0].set_ylabel('Confidence', fontsize = 20)
#  axs[0].grid(True)
#
#  axs[1].plot(combEpoch, D_fake)
#  axs[1].set_title(name + ': Discr Fake Confidence', fontsize = 20)
#  axs[1].set_xlabel('Iterations', fontsize = 20)
#  axs[1].set_ylabel('Confidence', fontsize = 20)
#  axs[1].grid(True)
#  plt.savefig(path + name + 'loss_D.jpg')
  
  
#Image display function
def tileCompFunc(inDict, runNameList, imgNum, subpath, preStr='val', save=False):
  rowSize = 5
  colSize = 5
  fig, axs = plt.subplots(rowSize, colSize, figsize=(15,15))
  
  stCol = 1
  stRow = 0
  
  #Get the image
  
  imPath = subpath + '/' + 'run_0' + '/saved_results/model_' + str(chosenModel) + '/' + preStr + '/'
  skPath = imPath + inDict[runNameList[0]][0][imgNum]
  bwPath = imPath+ inDict[runNameList[0]][1][imgNum]
  genPath = imPath + inDict[runNameList[0]][2][imgNum]
  
  skData = cv2.imread(skPath, 0)
  bwData = cv2.imread(bwPath, 0)
  
  #Set the top two as the original figures
  axs[0,0].imshow(skData, cmap='gray') #Sketch
  axs[0, 0].set_title('Sketch for ' + str(imgNum), fontsize = 12)
  
  axs[0,colSize-1].imshow(bwData, cmap='gray') #B&W
  axs[0, colSize-1].set_title('B&W for ' + str(imgNum), fontsize = 12)
  
  
  for runI in runNameList:
      
      imPath = subpath + '/' + runI + '/saved_results/model_' + str(chosenModel) + '/' + preStr + '/'
  
      
      genPath = imPath + inDict[runI][-1][imgNum]
      genData = cv2.imread(genPath, 0)
      
      axs[stCol, stRow].imshow(genData, cmap='gray')
      axs[stCol, stRow].set_title(runI, fontsize = 12)
      
      stRow += 1
      stCol += 1*(stRow % rowSize == 0)
      
      stRow = stRow % rowSize
      
  iterAx = axs.flatten()
  for ax in iterAx:
      ax.set_xticks([])
      ax.set_yticks([])
      
  
  if save:
      plt.savefig(resultPath + '/M' + str(chosenModel) + '_' + \
                  preStr + '_' + str(imgNum) + '.png')
      
      
      
      
#Image tiling comparing results of different models
def timeTiling(inDict, imgNum, fullPath, inRun, preStr='val', save=False):
  rowSize = 5
  colSize = 3
  fig, axs = plt.subplots(colSize, rowSize, figsize=(15,10))
  
  stCol = 1
  stRow = 0
  
  #Get the image
  #fullPath is the path up to the run and time tiling, should be ~
  #automated_tests_gcp/seed_x/run_x/saved_results/time_tiling/run_x_model_
  
  imPath = fullPath +  str(10) + '/' + preStr + '/'
  run_str = 'run_' + str(inRun)
  skPath = imPath + inDict[run_str][0][imgNum]
  bwPath = imPath+ inDict[run_str][1][imgNum]
  genPath = imPath + inDict[run_str][2][imgNum]
  
  skData = cv2.imread(skPath, 0)
  bwData = cv2.imread(bwPath, 0)
  
  #Set the top two as the original figures
  axs[0,0].imshow(skData, cmap='gray') #Sketch
  axs[0, 0].set_title('Sketch for ' + str(imgNum), fontsize = 12)
  
  axs[0, rowSize-1].imshow(bwData, cmap='gray') #B&W
  axs[0, rowSize-1].set_title('B&W for ' + str(imgNum), fontsize = 12)
  
  
  for currModel in modelList:
      
      imPath = fullPath + currModel + '/' + preStr + '/'
      genPath = imPath + inDict[run_str][-1][imgNum]
      
      genData = cv2.imread(genPath, 0)
      
      axs[stCol, stRow].imshow(genData, cmap='gray')
      axs[stCol, stRow].set_title('Model ' + currModel, fontsize = 12)
      
      stRow += 1
      stCol += 1*(stRow % rowSize == 0)
      
      stRow = stRow % rowSize
      
  iterAx = axs.flatten()
  for ax in iterAx:
      ax.set_xticks([])
      ax.set_yticks([])
      
  
  if save:
      plt.savefig(resultPath + '/time_tiling-run_' +  str(inRun) + '_' + \
                  preStr + '_' + str(imgNum) + '.png', dpi=500)
      
      
def calcHeuristic(inDict, runID, imgNum, subpath, preStr='val'):
    imPath = subpath + '/' + runID + '/saved_results/model_' + str(chosenModel) + '/' + preStr + '/'
    skPath = imPath + inDict[runID][0][imgNum]
    bwPath = imPath+ inDict[runID][1][imgNum]
    genPath = imPath + inDict[runID][2][imgNum]
    
    
    skData = np.float32(cv2.imread(skPath, 0))
    bwData = np.float32(cv2.imread(bwPath, 0))
    genData = np.float32(cv2.imread(genPath, 0))
    
    
    #Custom heuristic
#    heurVal = 1 - np.sum(np.abs(bwData - genData)) / np.sum(np.abs(bwData - skData))
    
    #L2 heuristic:
    H,W = bwData.shape
    heurVal = np.sum((bwData - genData)**2) / (H*W)
    return heurVal
    
      
  
          
   
    

#First, set the seed directory and make directories if they dont exist
#subproc will return 0 is successful, 1 if exists
####INPUT HERE####
txtIn = input('What seed do you want to test from? ')
seedNum = int(txtIn)

pathToData = 'path/to/data'
####INPUT HERE####
subSeedPath = 'automated_tests_gcp/seed_' + str(seedNum)
seedPath = 'automated_tests_gcp/seed_' + str(seedNum) + '/test_result'
checkPath = 'checkpoints/'
command = ['mkdir', seedPath]
a = subprocess.run(command)

compPath = seedPath + '/image_comparisons'
lossPath = seedPath + '/loss_plots'
isPath = seedPath + '/inception_plots'
resultPath = seedPath + '/saved_results'
miscPath = seedPath + '/misc'
tempPath = seedPath + '/temp'

if a.returncode == 0:
    #Means the directory has not been created yet
    print('Creating subdirectories...')
    
    command = ['mkdir', compPath]
    a = subprocess.run(command)
    
    command = ['mkdir', lossPath]
    a = subprocess.run(command)
    
    command = ['mkdir', isPath]
    a = subprocess.run(command)
    
    command = ['mkdir', resultPath]
    a = subprocess.run(command)
    
    command = ['mkdir', miscPath]
    a = subprocess.run(command)
    
    command = ['mkdir', tempPath]
    a = subprocess.run(command)
else:
    pass
    

####IMPLEMENTING RETRIEVAL OF PROPER MODELS AND IMAGES BELOW#####
#First, need to get a list of run directories
#Then need to retrieve the correct models
seedDirList = os.listdir(subSeedPath) 
runDirList = []

for i in seedDirList:
    if i[0:4] == 'run_':
        runDirList.append(i)
    else:
        pass
    
#Makes sure that the list is sorted alphabetically/by length with 2 sorts
runDirList.sort()
runDirList.sort(key=len) 
runNumList = [int(i[4:]) for i in runDirList]


#Now we choose the model to look at for all of the runs
####INPUT HERE####
#chosenModel = '5'
#For the latest model, use:
chosenModel = 50
valNum = 200
testNum = 200
####INPUT HERE####

modelStr = str(chosenModel) + '_net_G.pth'


###IMPLEMENTING INCEPTION SCORE CALCULATION HERE###
#First, use chosenModel to go through runDirList 
#Create checkpoint files for all things
#Run the test on all of the files inside using both validation and testing 
#Then, go inside the results folder, load each image, calculate inception score, and store in a list
#Then make some plots based on dimensionality and result

#First, remove the directory in checkpoints if it exists
#Then, copy desired model to the checkpoints as "latest_blahblah.pth"
#Then run the script and see the output in results

imgDictVal = {} 
imgDictTest = {} 

usrInRerun = input('Do you want rerun test/val image output [1 for yes]: ')

usrInDir = input('Do you want to delete the old directories if they pop up and replace them with the new images? [1 for yes]: ')

skipUserIn = input('Do you want to loop? [1 for loop, any other to skip to heuristic/loss: ')

if int(skipUserIn) == 1:
    
    for runPathIter in runDirList:
        
        #destroy/create new directory
        path = checkPath + runPathIter
        command = ['rm', '-r', path]
        subprocess.run(command)
        #again, returns 1 if unsuccessful and 0 if ok
        
        command = ['mkdir', path]
        subprocess.run(command)
        
        
        #copy over the desired model as latest
        try:
            srcPath = subSeedPath + '/' + runPathIter + '/checkpoints/' + modelStr
            sinkPath = checkPath + runPathIter + '/latest_net_G.pth'
            print('Copying over from ' + runPathIter)
            shutil.copy2(srcPath,sinkPath)
        except:
            print('ALERT, ALERT, RUN ' + runPathIter + ' DIDNT HAVE NEEDED MODEL')
            continue
        
        
        #Now, run the both the test and validation sets
        #For now, do 200 of each (remember, validation has like 700 and is different, test is from training)
    #    if int(usrInRerun) == 1 and runPathIter == 'run_0':
        if int(usrInRerun) == 1:
        
            command = ['rm', '-r', 'results/' + runPathIter]
            
            print('Running validation phase for ' + runPathIter + '...')
            subprocess.run(command)
            command = ['python', 'test.py', '--dataroot', pathToData, '--name', runPathIter, \
                       '--model', 'pix2pix', '--direction', 'AtoB', '--gpu_ids', 
                       '-1', '--num_test='+str(valNum), '--phase','val']
            subprocess.run(command)
            
            
            
            
            print('Running testing phase for ' + runPathIter + '...')
            command = ['python', 'test.py', '--dataroot', pathToData, '--name', runPathIter, \
                       '--model', 'pix2pix', '--direction', 'AtoB', '--gpu_ids', 
                       '-1', '--num_test='+str(testNum), '--phase','test']
            subprocess.run(command)
        
        
        #Now copy and parse over the files; copy their contents into lists to process
        testrunPath = subSeedPath + '/' + runPathIter + '/saved_results/model_' + str(chosenModel)
        command = ['mkdir', subSeedPath + '/' + runPathIter + '/saved_results']
        temp = subprocess.run(command)
        
        command = ['mkdir', testrunPath]
        temp = subprocess.run(command)
        
        if temp.returncode == 0:
            print('Created directory at ' + testrunPath)
        elif temp.returncode == 1:
            if int(usrInDir) == 1:
                command = ['rm', '-r', testrunPath]
                subprocess.run(command)
                
                command = ['mkdir', testrunPath]
                subprocess.run(command)
            else:
                print('Using old directory (not rewriting images)...')
                
        
        if temp.returncode == 0 or (temp.returncode == 1 and int(usrInDir) == 1):
            #First remove the DS_store file so we dont get any 
            srcPath = 'results/' + runPathIter + '/test_latest/images/'
            sinkPath = testrunPath + '/test'
            
    #        command = ['rm', srcPath + '.DS_Store']
    #        subprocess.run(command)
            
            shutil.copytree(srcPath,sinkPath)
            
            print('Test directory copied!')
            
            
            srcPath = 'results/' + runPathIter + '/val_latest/images/'
            sinkPath = testrunPath + '/val'
            
    #        command = ['rm', srcPath + '.DS_Store']
    #        subprocess.run(command)
            
            shutil.copytree(srcPath,sinkPath)
            
            print('Val directory copied!')
            
            
        
        #Now run extraction script to read individual files and store inside lists
        #Structure is as follows:
        #imgDict is a dictionary containing a list
        #list is [sketch img, black and white image, generated image]
        #each listentry is a list of the names of the images
        #Thus, for a given run, we can query the dictionary, and iterate through one of the lists
        path = testrunPath + '/val'
        dirList = os.listdir(path)
        dirList.sort()
        try:
            dirList.remove('.DS_Store')
        except:
            print('os_dir wasnt in the dir')
        
        #When sorted, order is: xxx_fake_B.png, xxx_real_A.png, xxx_real_B.png, etc
        skL = []
        bwL = []
        genL = []
        for i in range(len(dirList)//3):
            skL.append(dirList[i*3+1])
            bwL.append(dirList[i*3+2])
            genL.append(dirList[i*3])
            
        imgDictVal[runPathIter] = [skL,bwL,genL]
            
            
        path = testrunPath + '/test'
        dirList = os.listdir(path)
        dirList.sort()
        try:
            dirList.remove('.DS_Store')
        except:
            print('os_dir wasnt in the dir')
        
        #When sorted, order is: xxx_fake_B.png, xxx_real_A.png, xxx_real_B.png, etc
        skL = []
        bwL = []
        genL = []
        for i in range(len(dirList)//3):
            skL.append(dirList[i*3+1])
            bwL.append(dirList[i*3+2])
            genL.append(dirList[i*3])
            
        imgDictTest[runPathIter] = [skL,bwL,genL]
        
    
    
##FINISHED WITH RUN DIRECTORY LOOP!##
    
####NOW, CALCULATE IS SCORES!!###
    ##COME BACK TO THIS, DONT KNOW HOW TO DO INCEPTION SCORE!!!!
#ISDictVal = {} #creating a dictionary for the inception scores
#ISDictTest = {}
#for runD in runDirList:
#    valPath = testrunPath + '/val/'
#    
#    currList = imgDictVal[runD]
#    skPath = currList[0]
#    bwPath = currList[1]
#    genPath = currList[2]
#    
#    for i in range(len(skPath)):
#        skIm = cv2.imread(valPath + skPath, 0)
#        bwIm = cv2.imread(valPath + bwPath, 0)
#        genIm = cv2.imread(valPath + genPath, 0)
    
    
##NOW IMPLEMENT IMAGE CONCTENATION##
#Make sure to replace BOTH preStr variables and the imgDict for the tiling function!
#dispImList = [7,69,100,138,77,1,89,193]
    dispImList = [7,69,99,77,1,89]

ifRun = input('Do you want to run tile images? [1 for yes]: ')

if int(ifRun) == 1:
    ifSave = input('Do you want to save the tile images? [1 for yes]: ')
    saveIn = False
    if int(ifSave) == 1:
        saveIn = True
    
    for dispIm in dispImList:
        tileCompFunc(imgDictVal, runDirList, dispIm, subSeedPath, preStr='val', save=saveIn)
    
    for dispIm in dispImList:
        tileCompFunc(imgDictTest, runDirList, dispIm, subSeedPath, preStr='test', save=saveIn)
        

##IMPLEMENT CUSTOM HEURISTIC FUNCTION BELOW##
#Test on run0
print('Calculating heuristic index, val...')
inTxt = input('Do you want to calc heuristic index? [1 for yes]: ')
if int(inTxt) == 1:
    
    plt.figure(0)
    for runID in runDirList:
        heurArr = []
        
        try:
            for i in range(len(imgDictVal[runID][0])):
                heurArr.append(calcHeuristic(imgDictVal, runID, i, subSeedPath, preStr='val'))
            plt.plot(heurArr)
            print('Mean for ' + runID + ' = ' + str(np.mean(heurArr)))
        except:
            print('ALERT ALERT COULDNT RETRIEVE INFO FROM ' + runID)
            continue
        
        
        
        
    print('Calculating heuristic index, TEST...')
    
    
    plt.figure(1)
    for runID in runDirList:
        heurArr = []
        
        for i in range(len(imgDictTest[runID][0])):
            heurArr.append(calcHeuristic(imgDictTest, runID, i, subSeedPath, preStr='test'))
        plt.plot(heurArr)
        print('Mean for ' + runID + ' = ' + str(np.mean(heurArr)))
        
        



##IMPLEMENT LOSS PLOTTING##
inTxt = input('Do you want to plot losses? [1 for yes]: ')
if int(inTxt) == 1:
    print('Plotting losses...')
    runUserIn = input('Which run to plot for? ')
    
    if runUserIn == 'all':
        for runPathIter in runDirList:
            try:
                runPath = subSeedPath + '/' + runPathIter + '/loss'
                G_GAN, G_loss, D_real, D_fake, combEpoch = lossDataLoad(runPath)
                
                plotName = runPathIter + '-'
                lossPlotFunc(G_GAN, G_loss, D_real, D_fake, combEpoch, lossPath + '/', plotName)
            except:
                print(runPathIter + ' didnt have loss log')
            
    else:
    
        runPath = subSeedPath + '/run_' + runUserIn + '/loss'
        G_GAN, G_loss, D_real, D_fake, combEpoch = lossDataLoad(runPath)
        
        plotName = 'run_' + runUserIn + '-'
        lossPlotFunc(G_GAN, G_loss, D_real, D_fake, combEpoch, lossPath + '/', plotName)\
        
        

        
    
##IMPLEMENT SIDE-BY-SIDE PLOTTING##
    
inTxt = input('Do you want tile image transform flow? [1 for yes]: ')
if int(inTxt) == 1:
    print('Tiling images...')
    runUserIn = input('Which run to plot for? ')
    
    
    ##INPUT HERE
    maxModel = 90
    modelSpacing = 10
    
    tileNum = 20
    
    val = True
    imgVal = 10
    ##INPUT HERE
    
    #create model list with strings (so I dont have to do it later with paths)
    temp = list(np.arange(modelSpacing,maxModel + modelSpacing, modelSpacing))
    modelList = [str(i) for i in temp]
    
    #Define other variables we'll need for these operations
    
    
    
    

    imgGenUser = input('Run the tiling image generation again? [1 to run again]: ')
    if int(imgGenUser) == 1:
        
        #First make the directory that we'll be running the tiling in
        savingPath = subSeedPath + '/run_' + runUserIn + '/saved_results/time_tiling'
        command = ['mkdir', savingPath]
        subprocess.run(command)
        
        for currModel in modelList:
            
            print('Running model ' + currModel + '...')
            
            currModelPath = savingPath + '/run_'+ runUserIn + '_model_' + currModel
            currModelCheckpoint = 'time_tiling-run_' + runUserIn + '_model_' + currModel
            
            
            #destroy/create new directory
            path = checkPath + currModelCheckpoint
            command = ['rm', '-r', path]
            subprocess.run(command)
            #again, returns 1 if unsuccessful and 0 if ok
            
            command = ['mkdir', path]
            subprocess.run(command)
            
            
            #copy over the desired model as latest
            srcPath = subSeedPath + '/run_' + runUserIn + '/checkpoints/' + currModel + '_net_G.pth'
            sinkPath = 'checkpoints/' + currModelCheckpoint + '/latest_net_G.pth'
            print('Copying over from run_' + runUserIn)
            shutil.copy2(srcPath,sinkPath)
            
            
            if val:
                testPhase = 'val'
                testPhasePath = '/val_latest'
            else:
                testPhase = 'test'
                testPhasePath = '/test_latest'
            
        
            
            command = ['rm', '-r', 'results/' + currModelCheckpoint]
            subprocess.run(command)
            
            
            command = ['python', 'test.py', '--dataroot', pathToData, '--name', currModelCheckpoint, \
                       '--model', 'pix2pix', '--direction', 'AtoB', '--gpu_ids', 
                       '-1', '--num_test='+str(tileNum), '--phase',testPhase]
            subprocess.run(command)
            
         

            #Create new directory AT THE SAVE PATH FOR 
            command = ['mkdir', currModelPath]
            subprocess.run(command)
            
            
            
            #Now copy back images that have been created
            
            srcPath = 'results/' + currModelCheckpoint + testPhasePath + '/images/'
            sinkPath = currModelPath + '/' + testPhase 
            
            command = ['rm', '-r', sinkPath]
            subprocess.run(command)
            
            shutil.copytree(srcPath,sinkPath)
            
            print('Val directory copied!')
        
        print('Finished Generating Images!')
    
    #Now grab the images and tile em!
#    for currModel in modelList:
        
        
    ifSave = input('Do you want to save the time tile images? [1 for yes]: ')
    saveIn = False
    if int(ifSave) == 1:
        saveIn = True
        
    continueRunning = True
    imgVal
        
        
        
    while continueRunning:
    
        #automated_tests_gcp/seed_x/run_x/saved_results/time_tiling/run_x_model_
        #def timeTiling(inDict, imgNum, fullPath, inRun, preStr='val', save=False):
        fullPathIn = subSeedPath + '/run_' + runUserIn + \
                   '/saved_results/time_tiling/run_' + runUserIn + '_model_'
       
        timeTiling(imgDictVal, imgVal, fullPathIn, int(runUserIn), preStr=testPhase, save=saveIn)
#        timeTiling(imgDictTest, imgVal, fullPathIn, int(runUserIn), preStr=testPhase, save=saveIn)
        
        ifRunagain = input('Run again? [1 for yes]: ')
        
        if int(ifRunagain) == 1:
            continueRunning = True
            whichImg = input('Which image? ')
            imgVal = int(whichImg)
        else:
            continueRunning = False
            
            
#Best testing images in first 20:
#Val:
            #0, 5, 8, 12, 19
#Test:
            #1, 9, 11, 13, 17, 18
            
##Plotting loss L1 loss *ASSUMING USING 1 RUN*
inTxt = input('Do you want to plot L1 curves with same run? [1 for yes]: ')
if int(inTxt) == 1:
    print('Plotting...')
    
    for i in range(tileNum):
        
        
        
        
            
            
            
            

    
    
    
        
        
        
        
        
            
    
            



    
    

#python test.py --dataroot path/to/data --name run_0 --model pix2pix --direction AtoB --gpu_ids -1 --num_test=300 --phase val
    
    
    







    

    
    




#####Ref code below for plotting stuff
#
#ershStr = 'test0_nom_iter1/ershov_lossFolder_2'
#fullString = '/content/drive/My Drive/CS231N/' + ershStr
#
#G_GAN, G_loss, D_real, D_fake, combEpoch = lossDataLoad(fullString)
#
#print(G_loss)
#
#lossPlotFunc(G_GAN, G_loss, D_real, D_fake, combEpoch)


