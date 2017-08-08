
#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""
   This loads the LUNA16 mhd files (3D images), extracts the transverse patches (64x64)
   around the candidate positions, and then saves those patches to a subdirectory.
   In another script we'll take those patches and run them through a modified
   VGG model to see if we can correctly classify nodule (class 1) from
   non-nodule (class 0).
"""

import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
import ntpath
from neon.util.argparser import NeonArgparser

import logging

# parse the command line arguments
parser = NeonArgparser(__doc__)

parser.add_argument("--subset", default='subset0',
                    help='LUNA16 subset directory to process')

args = parser.parse_args()

# To get the original LUNA16 MHD data:
# wget https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AAAs2wbJxbNM44-uafZyoMVca/subset5.zip
# The files are 7-zipped. Regular linux unzip won't work to uncompress them. Use 7za instead.
# 7za e subset5.zip

DATA_DIR = '/mnt/data/medical/luna16/'
SUBSET = args.subset
cand_path = 'CSVFILES/candidates_with_annotations.csv'  # Candidates file tells us the centers of the ROI for candidate nodules

# Set up logging
logger = logging.getLogger(__name__)
hdlr = logging.FileHandler('all_'+SUBSET+'.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

def extractCandidates(img_file):
    
    # Get the name of the file
    subjectName = ntpath.splitext(ntpath.basename(img_file))[0]  # Strip off the .mhd extension
    
    # Read the list of candidate ROI
    dfCandidates = pd.read_csv(DATA_DIR+cand_path)
    
    
    numCandidates = dfCandidates[dfCandidates['seriesuid']==subjectName].shape[0]
    print('Subject {}: There are {} candidate nodules in this file.'.format(subjectName, numCandidates))
    
    numNonNodules = sum(dfCandidates[dfCandidates['seriesuid']==subjectName]['class'] == 0)
    numNodules = sum(dfCandidates[dfCandidates['seriesuid']==subjectName]['class'] == 1)
    print('{} are true nodules (class 1) and {} are non-nodules (class 0)'.format(numNodules, numNonNodules))
    
    # Read if the candidate ROI is a nodule (1) or non-nodule (0)
    candidateValues = dfCandidates[dfCandidates['seriesuid']==subjectName]['class'].values
    
    # Get the world coordinates (mm) of the candidate ROI center
    worldCoords = dfCandidates[dfCandidates['seriesuid']==subjectName][['coordX', 'coordY', 'coordZ']].values
    
    # Use SimpleITK to read the mhd image
    itkimage = sitk.ReadImage(img_file)
    
    # Get the real world origin (mm) for this image
    originMatrix = np.tile(itkimage.GetOrigin(), (numCandidates,1))  # Real world origin for this image (0,0)
    
    # Subtract the real world origin and scale by the real world (mm per pixel)
    # This should give us the X,Y,Z coordinates for the candidates
    candidatesPixels = (np.round(np.absolute(worldCoords - originMatrix) / itkimage.GetSpacing())).astype(int)
       
    candidatePatches = []
    
    imgAll = sitk.GetArrayFromImage(itkimage) # Read the image volume

    valueArray = []

    for candNum in range(numCandidates):
        
        #print('Extracting candidate patch #{}'.format(candNum))
        candidateVoxel = candidatesPixels[candNum,:]
        xpos = int(candidateVoxel[0])
        ypos = int(candidateVoxel[1])
        zpos = int(candidateVoxel[2])
        
        # Need to handle the candidates where the window would extend beyond the image boundaries
        windowSize = 64  # Center a 64 pixel by 64 pixel patch around the candidate position
        x_lower = np.max([0, xpos - windowSize//2])  # Return 0 if position off image
        x_upper = np.min([xpos + windowSize//2, itkimage.GetWidth()]) # Return  maxWidth if position off image
        
        y_lower = np.max([0, ypos - windowSize//2])  # Return 0 if position off image
        y_upper = np.min([ypos + windowSize//2, itkimage.GetHeight()]) # Return  maxHeight if position off image
        
        z_lower = np.max([0, zpos - windowSize//2])  # Return 0 if position off image
        z_upper = np.min([zpos + windowSize//2, itkimage.GetDepth()]) # Return  maxHeight if position off image
         
        skipPatch = False
        if ((xpos - windowSize//2) < 0) | ((xpos + windowSize//2) > itkimage.GetWidth()):
            logger.info('img file {} off x for candidate {}, label {}'.format(img_file, candNum, candidateValues[candNum]))
            skipPatch = True

        if ((ypos - windowSize//2) < 0) | ((ypos + windowSize//2) > itkimage.GetHeight()):
            logger.info('img file {} off y for candidate {}, label {}'.format(img_file, candNum, candidateValues[candNum]))
            skipPatch = True

        # SimpleITK is x,y,z. Numpy is z, y, x.
        imgPatch = imgAll[zpos, y_lower:y_upper, x_lower:x_upper]
        
        #imgPatch = imgAll[zpos, :, :]
          
        # Normalize to the Hounsfield units
        imgPatchNorm = normalizePlanes(imgPatch)
        
        if not skipPatch:
            candidatePatches.append(imgPatchNorm)  # Append the candidate image patches to a python list
            valueArray.append(candidateValues[candNum])

    return candidatePatches, valueArray

"""
Normalize pixel depth into Hounsfield units (HU)

This tries to get all pixels between -1000 and 400 HU.
All other HU will be masked.
Then we normalize pixel values between 0 and 1.

"""
def normalizePlanes(npzarray):
     
    maxHU = 400.
    minHU = -1000.
 
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray


from scipy.misc import toimage

"""
Save the image patches for a given data file
"""
# We need to save the array as an image.
# This is the easiest way. Matplotlib seems to like adding a white border that is hard to kill.
def SavePatches(manifestFilename, img_file, patchesArray, valuesArray):
    
    saveDir = ntpath.dirname(img_file) + '/patches_ALL'

    try:
        os.stat(saveDir)
    except:
        os.mkdir(saveDir) 

    with open(manifestFilename, 'a') as f:  # Write to the manifest file for aeon loader

        subjectName = ntpath.splitext(ntpath.basename(img_file))[0]
        

        print('Saving image patches for file {}/{}.'.format(SUBSET, subjectName))
        for i in range(len(valuesArray)):

            
            #print('\r{} of {}'.format(i+1, len(valuesArray))),
            im = toimage(patchesArray[i])

            pngName = saveDir + '/{}_{}_{}.png'.format(subjectName, i, valuesArray[i])
            im.save(pngName)

            f.write('{},label_{}.txt\n'.format(pngName, valuesArray[i]))

        f.close()

        print('{}: Finished {}\n'.format(SUBSET, subjectName))


"""
Loop through all .mhd files within the data directory and process them.
"""

# Reset the manifest file to empty
manifestFilename = 'manifest_{}_ALL.csv'.format(SUBSET)
f = open(manifestFilename, 'w')
f.close()

for root, dirs, files in os.walk(DATA_DIR+SUBSET):
    
    for file in files:
        
        if (file.endswith('.mhd')) & ('__MACOSX' not in root):  # Don't get the Macintosh directory
            

            img_file = os.path.join(root, file)

            patchesArray, valuesArray = extractCandidates(img_file)   
             
            SavePatches(manifestFilename, img_file, patchesArray, valuesArray)
                
                



