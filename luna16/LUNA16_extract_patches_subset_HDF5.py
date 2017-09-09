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
Loads the LUNA16 dataset (subset) and extracts 
a 3D image tensor for the region of interest.
"""

import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
import ntpath
from neon.util.argparser import NeonArgparser
import h5py


# parse the command line arguments
parser = NeonArgparser(__doc__)

parser.add_argument('--subset', default=0,
                    help='LUNA16 subset directory to process')
parser.add_argument('--augment', action='store_true',
                    help='Augment class 1 and subsample class 0 for better class balance.')

parser.add_argument('--exclude', action='store_true',
                    help='Exclude this subset.')

args = parser.parse_args()


# To get the original LUNA16 MHD data:
# wget https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AAAs2wbJxbNM44-uafZyoMVca/subset5.zip
# The files are 7-zipped. Regular linux unzip won't work to uncompress them. Use 7za instead.
# 7za e subset5.zip

DATA_DIR = '/mnt/data/medical/luna16/'
EXCLUDE_DIR = args.exclude # include all directories except the subset one; otherwise just include the subset
SUBSET = 'subset{}'.format(args.subset)

if EXCLUDE_DIR:
    cand_path = 'CSVFILES/candidates.csv'  # Candidates file tells us the centers of the ROI for candidate nodules
else:
    cand_path = 'CSVFILES/candidates_V2.csv'  # Candidates file tells us the centers of the ROI for candidate nodules

window_width = 32 # This is really the half width so window will be double this width
window_height = 32 # This is really the half height so window will be double this height
window_depth = 5 # This is really the half depth so window will be double this depth
num_channels = 1

def find_bbox(center, origin,  
              mask_width, mask_height, mask_depth,
              depth, height, width):
    '''
    Center : center region of interest -- list of coordinates x,y,z
    origin = x,y,z mm np.array
    '''
    
    # TODO:  The height and width seemed to be switched. 
    # This works but needs to be simplified. It's probably due to SimpleITK versus Numpy transposed indicies.

    # NOTE: The np.abs are needed because some of the CT scans have the axes flipped. 
    # The proper method would be to use simple ITK's GetDirection to determine the
    # flip and do a matrix transformation. However, the LUNA16 authors promote just
    # taking the absolute value since it is only a few files that are affected
    # and it's a simple axis flip.
    x = int(np.round(np.abs(center[0] - origin[0])))
    y = int(np.round(np.abs(center[1] - origin[1])))
    z = int(np.round(np.abs(center[2] - origin[2])))

    left = x - mask_width
    if left <= 0:
        pad_left = -left + 1
        left = 0
    else:
        pad_left = 0

    right = x + mask_width
    if right > width:
        pad_right = right - width
        right = width
    else:
        pad_right = 0

    down = y - mask_height
    if down <= 0:
        pad_down = -down + 1
        down = 0
    else:
        pad_down = 0

    up = y + mask_height
    if up > height:
        pad_up = up - height
        up = height
    else:
        pad_up = 0
    
    bottom = z - mask_depth
    if bottom <= 0:
        pad_bottom = -bottom + 1
        bottom = 0
    else:
        pad_bottom = 0

    top = z + mask_depth
    if top > depth:
        pad_top = top - depth 
        top = depth
    else:
        pad_top = 0


    bbox = [[down, up], [left, right], [bottom, top]]
    padding = [[pad_down, pad_up], [pad_left, pad_right], [pad_bottom, pad_top]]
    
    return bbox, padding

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
This tries to get all pixels between -1000 and 2000 HU.
All other HU will be masked.
Then we normalize pixel values between 0 and 1.
"""
def normalizePlanes(npzarray):
     
    maxHU = 2000.
    minHU = -1000.
 
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

def extract_candidates(img_file):

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

    return candidateValues, worldCoords

def extract_tensor(img_array, worldCoords, origin, spacing):

    '''
    Extracts 3D tensor around the center for the ROI.
    The sticky point here is the order of the axes. Numpy is z,y,x and SimpleITK is x,y,z.
    I've found it very difficult to keep the order correct when going back and forth,
    but this code seems to pass the sanity checks.
    '''
    
    slice_z, height, width = img_array.shape        
    
    # This is the real world x,y,z coordinates of possible nodule (in mm)
    candidate_x = worldCoords[0]
    candidate_y = worldCoords[1]
    candidate_z = worldCoords[2]
    
    center = np.array([candidate_x, candidate_y, candidate_z])   # candidate center
    voxel_center = np.rint((center-origin)/spacing).astype(int)  # candidate center in voxel space (still x,y,z ordering)

    # Calculates the bounding box for desired position
    bbox, pad_needed = find_bbox(center, origin, window_width, window_height, window_depth,
                              slice_z, height, width)
        
    if (np.sum(pad_needed) == 0):

        # ROI volume tensor
        # D x H x W
        img = normalizePlanes(img_array[bbox[2][0]:bbox[2][1], 
                                        bbox[0][0]:bbox[0][1], 
                                        bbox[1][0]:bbox[1][1]])
        
        #img = img.transpose([1, 2, 0])  # Height, Width, Depth

        # Then we need to flatten the array to a single vector (1, C*H*W*D)
        imgTensor = img.ravel().reshape(1,-1)


    else:
        imgTensor = None

    return imgTensor   


"""
Loop through all .mhd files within the data directory and process them.
"""
USE_AUGMENTATION = args.augment

if EXCLUDE_DIR:
    excludeName = 'except_'
    DATA_DIR_DESCEND = DATA_DIR
else:
    excludeName = ''
    DATA_DIR_DESCEND = DATA_DIR + SUBSET

if USE_AUGMENTATION:
    outFilename = DATA_DIR + 'luna16_roi_' + excludeName + SUBSET + '_augmented.h5'
else:
    outFilename = DATA_DIR + 'luna16_roi_' + excludeName + SUBSET + '_ALL.h5'

firstTensor = True


valuesArray = []
posArray = []
tensorShape = num_channels*(window_height*2)*(window_width*2)*(window_depth*2) # CxHxWxD


def writeToHDF(img, dset, val, valuesArray, posArray, worldCoords, fileName):

    # HDF5 allows us to dynamically resize the dataset
    row = dset.shape[0] # How many rows in the dataset currently?
    dset.resize(row+1, axis=0)   # Add one more row (i.e. new ROI)
    dset[row, :] = imgTensor  # Append the new row to the dataset

    valuesArray.append(val)

    subjectName = ntpath.splitext(ntpath.basename(fileName))[0]  # Strip off the .mhd extension

    posArray.append([subjectName, worldCoords[0], worldCoords[1], worldCoords[2]])

'''
Main
'''

with h5py.File(outFilename, 'w') as df:  # Open hdf5 file for writing our DICOM dataset

    for root, dirs, files in os.walk(DATA_DIR_DESCEND, topdown=True):

        # If EXCLUDE_DIR true, then include all directories except SUBSET
        # Otherwise, exclude all directories except SUBSET
        if EXCLUDE_DIR:
            dirs[:] = [d for d in dirs if d not in SUBSET]  # Exclude the subset

        for file in files:
            
            if (file.endswith('.mhd')) & ('__MACOSX' not in root):  # Don't get the Macintosh directory
                
                img_file = os.path.join(root, file)
                print(img_file)

                candidateValues, worldCoords = extract_candidates(img_file)

                # Load the CT scan (3D .mhd file)
                itk_img = sitk.ReadImage(img_file)  # indices are x,y,z (note the ordering of dimesions)

                # Normalize the image spacing so that a voxel is 1x1x1 mm in dimension
                itk_img = normalize_img(itk_img)

                # SimpleITK keeps the origin and spacing information for the 3D image volume
                img_array = sitk.GetArrayFromImage(itk_img) # indices are z,y,x (note the ordering of dimensions)
            
                numNegatives = 50

                for candidate_idx in range(candidateValues.shape[0]): # Iterate through all candidates

                    imgTensor = extract_tensor(img_array, worldCoords[candidate_idx, :], 
                                               np.array(itk_img.GetOrigin()), np.array(itk_img.GetSpacing()))

                    if (imgTensor is not None):

                        if (firstTensor): # For first value we need to create the dataset

                            dset = df.create_dataset('input', data=imgTensor, maxshape=[None, tensorShape])
                            valuesArray.append(candidateValues[candidate_idx])
                            posArray.append([ntpath.splitext(ntpath.basename(file))[0], worldCoords[candidate_idx, 0], worldCoords[candidate_idx,1],
                                             worldCoords[candidate_idx,2]])

                            firstTensor = False

                            if (candidateValues[candidate_idx] == 0):
                                numNegatives -= 1

                            continue   # Go back to the top

                        if not USE_AUGMENTATION:

                            writeToHDF(imgTensor, dset, candidateValues[candidate_idx], valuesArray,
                                posArray, worldCoords[candidate_idx, :], file)


                        elif (candidateValues[candidate_idx] == 0) & (numNegatives > 0):

                            # Flip a coin to determine if we keep this negative sample
                            if (np.random.random_sample() > 0.5):


                                writeToHDF(imgTensor, dset, candidateValues[candidate_idx], valuesArray,
                                    posArray, worldCoords[candidate_idx, :], file)

                                numNegatives -= 1

                        
                        elif (candidateValues[candidate_idx] == 1):

                            writeToHDF(imgTensor, dset, candidateValues[candidate_idx], valuesArray,
                                posArray, worldCoords[candidate_idx, :], file)

                            # #img = imgTensor.reshape(num_channels, window_height*2, window_width*2, window_depth*2)
                            # img = imgTensor.reshape(num_channels, window_depth*2, window_height*2, window_height*2)

                        
                            # imgFlipH = img[:,::-1,:,:] # Flip height dimension
                            # imgFlipW = img[:,:,::-1,:] # Flip width dimension
                            # imgFlipD = img[:,:,:,::-1] # Flip depth dimension

                            # writeToHDF(imgFlipH, dset, candidateValues[candidate_idx], valuesArray,
                            #     posArray, worldCoords[candidate_idx, :], file)

                            # writeToHDF(imgFlipW, dset, candidateValues[candidate_idx], valuesArray,
                            #     posArray, worldCoords[candidate_idx, :], file)

                            # writeToHDF(imgFlipD, dset, candidateValues[candidate_idx], valuesArray,
                            #     posArray, worldCoords[candidate_idx, :], file)


                            # # Augment positives by rotation 
                            # img90h = np.rot90(img.transpose([1,2,3,0])).transpose([3,0,1,2]) # Rotate 90 deg height dimension

                            # img90w = np.rot90(img.transpose([3,1,2,0])).transpose([3,1,2,0]) # Rotate 90 deg width dimension

                            # img90d = np.rot90(img.transpose([2,1,3,0])).transpose([3,1,0,2]) # Rotate 90 deg depth dimension

                            # writeToHDF(img90h, dset, candidateValues[candidate_idx], valuesArray,
                            #     posArray, worldCoords[candidate_idx, :], file)

                            # writeToHDF(img90w, dset, candidateValues[candidate_idx], valuesArray,
                            #     posArray, worldCoords[candidate_idx, :], file)

                            # writeToHDF(img90d, dset, candidateValues[candidate_idx], valuesArray,
                            #     posArray, worldCoords[candidate_idx, :], file)


    print('Writing shape and output to HDF5 file.')

    # Output attribute 'lshape' to let neon know the shape of the tensor.
    #df['input'].attrs['lshape'] = (num_channels, window_height*2, window_width*2, window_depth*2) # (Channel, Height, Width, Depth)
    
    # There's only one channel so let's see if we can shove the 3rd dimension into the channels
    # It won't be 3D convolution, but perhaps we'll get something out of it.
    #df['input'].attrs['lshape'] = (1, window_height*2, window_width*2, window_depth*2) # (Height, Width, Depth)
    
    df['input'].attrs['lshape'] = (window_depth*2, window_height*2, window_width*2) # (Height, Width, Depth)
    
    # Output the labels
    valuesArray = np.array(valuesArray)
    df.create_dataset('output', data=valuesArray.reshape(-1,1))
    df['output'].attrs['nclass'] = 2

    # Keep the positions in the HDF5 file
    df.create_dataset('position', data=np.array(posArray))
    
    num0 = len(np.where(valuesArray == 0)[0])
    num1 = len(np.where(valuesArray == 1)[0])

    print('# class 0 = {}, # class 1 = {}, total = {}, ratio = {:.2f}'.format(num0, num1, len(valuesArray), float(num0)/num1))
            
                