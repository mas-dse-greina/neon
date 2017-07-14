
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

# # DICOM Processing Script
# 
# *Summary:* Take DICOM images and process them for use in neon models.
# 
# NOTES:
# + [DICOM](https://en.wikipedia.org/wiki/DICOM) stands for "Digital Imaging and COmmunication in Medicine". It is the standard for all medical imaging from MRI to CT to Ultrasound to whatever. 
# + The standard was created so that devices manufacturers would have a common format for hospitals to integrate into their digital imaging systems. That's good for us because it gives us a standard way to load and process many types of medical images. If we can get a good pipeline going, then this script might be useful for any medical imaging study.
# + MRI and CT are stored as 2D slices which can be combined to form a 3D volume. So for any given patient we may have several hundred slice files that correspond to the same scan. This is different than most of our image data models because we need to incoporate all of these slices into one tensor. We definitely want to use the 3D volume because things like tumors and lesions (and other bad things that cause illness) don't limit themselves to a single 2D slice. Therefore, we've got to load this into our model as a 4D tensor (channel, height, width, depth). 
# + The slice thickness varies but is typically 1 to 3 mm. It depends on the type of study (MR/CT) and the parameters set at the start of the scan. Be sure to get a good handle on the height, width, and depth parameters of the DICOM files so that you are importing consistent tensors into the data loader. This is something I need to look into because ideally we'd like to standardize the tensors (for example, 1 mm x 1 mm x 1 mm voxels or something like that). 
# + The pixels are usually stored as uint16 precision (however I'm seeing several that are 32-bit too). I'm not sure if we need to change that to something with less precision. If we can keep the full 16-bit precision, then that would be preferable. There may in fact be anomalies that involve a very minor difference in the contrast between two adjacent regions. This is an open question for analysis.
# + Along with the actual pixel information, the DICOM file also contains lots of metadata, such as slice thickness, pixel resolution, image orientation, patient orientation, and type of study (i.e. MRI, CT, X-ray).
# + This assumes that your DICOM images are stored in the directory data_dir. Within data_dir there should be a separate folder for each patient's scans. The files usually end in the .dcm extension (e.g. "0a291d1b12b86213d813e3796f14b329.dcm" might be one slice for one patient). The SimpleITK library we use will load slices within a given patient's directory all at once into a 3D object.

import numpy as np
import glob, os


# ## SimpleITK for reading DICOM
# 
# [SimpleItk](http://www.simpleitk.org/) is an open-source library for reading and processing 3D models. It was particularly designed for medical imaging and was built on top of the Insight and Segmentation and Registration ([ITK](https://itk.org/)) toolkit sponsored by the National Library of Medicine.
# 
# What's nice about SimpleITK is that it has pre-built methods to read all of the DICOM slices into a 3D object and perform segmentation and morphological operations on that 3D object. I believe it automatically arranges the slices in the correct order. It also can handle compressed DICOM files.

#!pip install SimpleITK
import SimpleITK as sitk


# Parse the command line

from neon.util.argparser import NeonArgparser

parser = NeonArgparser(__doc__)

parser.add_argument('-out', '--outFilename', default='dicom_out.h5', help='Name of the output HDF5 file')

args = parser.parse_args()

outFilename = args.outFilename

if len(args.data_dir) == 0:
    data_dir = "/Volumes/data/tonyr/dicom/Lung CT/stage1"

patients = glob.glob(os.path.join(data_dir, '*')) # Get the folder names for the patients

numPatients = len(patients)  # Number of patients in the directory

if numPatients == 0:
    raise IOError('Directory ' + data_dir + ' not found or no files found in directory')
    


print('Found the following subfolders with DICOMs: {}'.format(patients))


# ## Now load the entire set of DICOM images into one large HDF5 file
# 
# Alternatively, we could do some other pre-processing of the images (e.g. normalizing them, aligning them, segmenting them) and then save to HDF5.
# 
# However, for right now let's just load the data without any other pre-processing.

import h5py
import ntpath


# ## The 'input' expects the array to be flattened!
# 
# The [HDF5Iterator](http://neon.nervanasys.com/index.html/_modules/neon/data/hdf5iterator.html#HDF5Iterator) expects to get a 2D array where the rows are each sample and the columns are each feature. So for a CxHxW set of images, the number of features should be the product of those 3 dimensions. 
# 
# I need to add a depth dimension (D) for the slices. So I'll have a 2D array that is # samples x (CxHxWxD).

def getImageTensor(patientDirectory):
    """
    Helper function for injesting all of the DICOM files for one study into a single tensor
    
    input: 'patientDirectory', the directory where the DICOMs for a single patient are stored
    outputs:
            imgTensor = a flattened numpy array (1, C*H*W*D)
            C = number of channels per pixel (1 for MR, CT, and Xray)
            H = number of pixels in height
            W = number of pixels in width
            D = number of pixels in depth
    """
    reader = sitk.ImageSeriesReader()  # Set up the reader object
    
    # Now get the names of the DICOM files within the directory
    filenamesDICOM = reader.GetGDCMSeriesFileNames(patientDirectory)
    reader.SetFileNames(filenamesDICOM)
    # Now execute the reader pipeline
    patientObject = reader.Execute()

    C = patientObject.GetNumberOfComponentsPerPixel() # There is just one color channel in the DICOM for CT and MRI
    H = patientObject.GetHeight()  # Height in pixels
    W = patientObject.GetWidth()   # Width in pixels
    #D = patientObject.GetDepth()  # Depth in pixels
    D = 128   # Let's limit to 128 for now - 
        
    # We need to tranpose the SimpleITK ndarray to the right order for neon
    # Then we need to flatten the array to a single vector (1, C*H*W*D)
    imgTensor = sitk.GetArrayFromImage(patientObject[:,:,:D]).transpose([1, 2, 0]).ravel().reshape(1,-1)
            
    return imgTensor, C, H, W, D


# ## Loop through the patient directory and load the DICOM tensors into HDF5 file
# 
# HDF5 allows for the dataset to be dynamically updated. So this should iteratively append new DICOM tensors to the HDF5 file. Otherwise, we'd have to load all of the files into memory and quickly run out of space.
# 
# **TODO (priority low):** I think there's a parallel way of writing to HDF5. I might be able to speed things up by having parallel threads to load different patients and append them to the HDF5.


with h5py.File(outFilename, 'w') as df:  # Open hdf5 file for writing our DICOM dataset

    for patientDirectory in patients[:1]:  # Start with the first patient to set up the HDF5 dataset

        patientID = ntpath.basename(patientDirectory) # Unique ID for patient

        print('({} of {}): Processing patient: {}'.format(1, numPatients, patientID))

        imgTensor, original_C, original_H, original_W, original_D = getImageTensor(patientDirectory)
        
        dset = df.create_dataset('input', data=imgTensor, maxshape=[None, original_C*original_H*original_W*original_D])
        
        # Now iterate through the remaining patients and append their image tensors to the HDF5 dataset
        
        for i, patientDirectory in enumerate(patients[1:]): # Now append the remaining patients
            
            print('({} of {}): Processing patient: {}'.format(i+2, numPatients, ntpath.basename(patientDirectory)))

            imgTensor, C, H, W, D = getImageTensor(patientDirectory)
            
            # Sanity check
            # Let's make sure that all dimensions are the same. Otherwise, we need to pad (?)
            assert(C == original_C)
            assert(H == original_H)
            assert(W == original_W)
            assert(D == original_D)
            
            # HDF5 allows us to dynamically resize the dataset
            row = dset.shape[0] # How many rows in the dataset currently?
            dset.resize(row+1, axis=0)   # Add one more row (i.e. new patient)
            dset[row, :] = imgTensor  # Append the new row to the dataset
           
        
    # Output attribute 'lshape' to let neon know the shape of the tensor.
    df['input'].attrs['lshape'] = (C, H, W, D) # (Channel, Height, Width, Depth)

    print('FINISHED. Output to HDF5 file: {}'.format(outFilename))

