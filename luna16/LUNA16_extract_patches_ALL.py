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
Loads the LUNA16 dataset (subset) and extracts the
transverse, coronal, and sagittal 2D slices for the candidate positions.

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

parser.add_argument("--subset", default='subset9',
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
hdlr = logging.FileHandler('subset_'+SUBSET+'.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

def make_mask(center,diam,z,
              width,height, depth, 
              spacing, origin,  
              mask_width=32, mask_height=32, mask_depth=32):
    '''
Center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    padMask = 5
    
    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+padMask)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-padMask])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+padMask])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-padMask]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+padMask])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # SPHERICAL MASK
    # Fill in 1 within sphere around nodule 
#     for v_x in v_xrange:
#         for v_y in v_yrange:
#             p_x = spacing[0]*v_x + origin[0]
#             p_y = spacing[1]*v_y + origin[1]
#             if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
#                 mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
               
 	# RECTANGULAR MASK
    for v_x in v_xrange:
        for v_y in v_yrange:
            
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            
            if ((p_x >= (center[0] - mask_width)) &
                (p_x <= (center[0] + mask_width)) & 
                (p_y >= (center[1] - mask_height)) &
                (p_y <= (center[1] + mask_height))):
                
                mask[int((np.abs(p_y-origin[1]))/spacing[1]),int((np.abs(p_x-origin[0]))/spacing[0])] = 1.0
            
    
    # TODO:  The height and width seemed to be switched. 
    # This works but needs to be simplified. It's probably due to SimpleITK versus Numpy transposed indicies.

    # NOTE: The np.abs are needed because some of the CT scans have the axes flipped. 
    # The proper method would be to use simple ITK's GetDirection to determine the
    # flip and do a matrix transformation. However, the LUNA16 authors promote just
    # taking the absolute value since it is only a few files that are affected
    # and it's a simple axis flip.
    left = np.max([0, np.abs(center[0] - origin[0]) - mask_width]).astype(int)
    right = np.min([width, np.abs(center[0] - origin[0]) + mask_width]).astype(int)
    down = np.max([0, np.abs(center[1] - origin[1]) - mask_height]).astype(int)
    up = np.min([height, np.abs(center[1] - origin[1]) + mask_height]).astype(int)
    
    top = np.min([depth, np.abs(center[2] - origin[2]) + mask_depth]).astype(int)
    bottom = np.max([0, np.abs(center[2] - origin[2]) - mask_depth]).astype(int)
    
    bbox = [[down, up], [left, right], [bottom, top]]
    
    return mask, bbox

def normalize_img(img):
    
    '''
    Sets the MHD image to be approximately 1.0 mm voxel size
    
    https://itk.org/ITKExamples/src/Filtering/ImageGrid/ResampleAnImage/Documentation.html
    '''
    new_x_size = int(img.GetSpacing()[0]*img.GetWidth())  # Number of pixels you want for x dimension
    new_y_size = int(img.GetSpacing()[1]*img.GetHeight()) # Number of pixels you want for y dimension
    new_z_size = int(img.GetSpacing()[2]*img.GetDepth())  # Number of pixels you want for z dimesion
    new_size = [new_x_size, new_y_size, new_z_size]
    
#     new_spacing = [old_sz*old_spc/new_sz  for old_sz, old_spc, new_sz in zip(img.GetSize(), img.GetSpacing(), new_size)]

    new_spacing = [1,1,1]  # New spacing to be 1.0 x 1.0 x 1.0 mm voxel size
    interpolator_type = sitk.sitkLinear

    return sitk.Resample(img, new_size, sitk.Transform(), interpolator_type, img.GetOrigin(), new_spacing, img.GetDirection(), 0.0, img.GetPixelIDValue())
    
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

from scipy.misc import toimage, imrotate
"""
Save the image patches for a given data file
"""
# We need to save the array as an image.
# This is the easiest way. Matplotlib seems to like adding a white border that is hard to kill.
def SavePatches(manifestFilename, img_file, patchesArray_trans, patchesArray_sag, patchesArray_cor, valuesArray):
    
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
            im = toimage(patchesArray_trans[i])

            pngName = saveDir + '/{}_{}_{}.png'.format(subjectName, i, valuesArray[i])
            im.save(pngName)

            f.write('{},label_{}.txt\n'.format(pngName, valuesArray[i]))

                
        f.close()

        print('{}: Finished {}\n'.format(SUBSET, subjectName))

def extract_candidates(img_file):

	'''
	Extracts 2D patches from the 3 planes (transverse, coronal, and sagittal).

	The sticky point here is the order of the axes. Numpy is z,y,x and SimpleITK is x,y,z.
	I've found it very difficult to keep the order correct when going back and forth,
	but this code seems to pass the sanity checks.
	'''
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
	worldCoords = dfCandidates[dfCandidates['seriesuid']==subjectName][['coordX', 'coordY', 'coordZ', 'diameter_mm']].values

	# Load the CT scan (3D .mhd file)
	itk_img = sitk.ReadImage(img_file)  # indices are x,y,z (note the ordering of dimesions)

	# Normalize the image spacing so that a voxel is 1x1x1 mm in dimension
	itk_img = normalize_img(itk_img)

	# SimpleITK keeps the origin and spacing information for the 3D image volume
	img_array = sitk.GetArrayFromImage(itk_img) # indices are z,y,x (note the ordering of dimesions)
	slice_z, height, width = img_array.shape        
	origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm) - Not same as img_array
	spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coordinates (mm)

	valueArray = []
	candidatePatches_trans = []
	candidatePatches_sag = []
	candidatePatches_cor = []

	for candidate_idx in range(numCandidates): # Iterate through all candidates

		# This is the real world x,y,z coordinates of possible nodule (in mm)
		candidate_x = worldCoords[candidate_idx, 0]
		candidate_y = worldCoords[candidate_idx, 1]
		candidate_z = worldCoords[candidate_idx, 2]
		diam = worldCoords[candidate_idx, 3]  # Only defined for true positives
		if (np.isnan(diam)):  # False positives are not labeled with diameter
			diam = 10

		mask_width = 32 # This is really the half width so window will be double this width
		mask_height = 32 # This is really the half height so window will be double this height
		mask_depth = 32 # This is really the half depth so window will be double this depth

		center = np.array([candidate_x, candidate_y, candidate_z])   # candidate center
		voxel_center = np.rint((center-origin)/spacing).astype(int)  # candidate center in voxel space (still x,y,z ordering)

		# Calculates the bounding box (and ROI mask) for desired position
		mask, bbox = make_mask(center, diam, voxel_center[2]*spacing[2]+origin[2],
		                       width, height, slice_z, spacing, origin, 
		                       mask_width, mask_height, mask_depth)
		    
		# Transverse slice 2D view - Y-X plane
		# Confer with https://en.wikipedia.org/wiki/Anatomical_terms_of_location#Planes
		img_transverse = normalizePlanes(img_array[voxel_center[2], 
		                                           bbox[0][0]:bbox[0][1], 
		                                           bbox[1][0]:bbox[1][1]])

		# Sagittal slice 2D view - Z-Y plane
		img_sagittal = normalizePlanes(img_array[bbox[2][0]:bbox[2][1], 
		                                         bbox[0][0]:bbox[0][1], 
		                                         voxel_center[0]])

		# Coronal slice 2D view - Z-X plane
		img_coronal = normalizePlanes(img_array[bbox[2][0]:bbox[2][1], 
		                                        voxel_center[1], 
		                                        bbox[1][0]:bbox[1][1]])

		skipPatch = False
		if not skipPatch:
		    candidatePatches_trans.append(img_transverse) 
		    candidatePatches_cor.append(img_coronal) 
		    candidatePatches_sag.append(img_sagittal) 
		    valueArray.append(candidateValues[candidate_idx])

	return candidatePatches_trans, candidatePatches_sag, candidatePatches_cor, valueArray

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

            patchesArray_trans, patchesArray_sag, patchesArray_cor,  valuesArray = extract_candidates(img_file)   
             
            SavePatches(manifestFilename, img_file, patchesArray_trans, patchesArray_sag, patchesArray_cor, valuesArray)
                
