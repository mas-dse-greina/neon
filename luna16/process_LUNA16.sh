#!/bin/bash
# Usage:  ./process_LUNA16.sh 40 5 50
# Run entire cross-validation analysis using patch size 40 pixels, patch depth 5 pixels, and 50 epochs for training

ARG1=${1:-50}  # Size of the patch (in pixels)
ARG2=${2:-5}   # Depth of the patch (in pixels)
ARG3=${3:-40}  # Number of epochs to train

echo 'Using patch parameters: size ' ${ARG1} 'pixels, depth = ' ${ARG2} 'pixels' 
echo 'Training for ' ${ARG3} 'epochs'

# Extract the ROI patches to HDF dataset files
python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 0 &
python LUNA16_extract_patches_subset_HDF5.py --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 0&
python LUNA16_extract_patches_subset_HDF5.py --patch_size ${ARG1} --patch_depth ${ARG2} --subset 0 &

python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 1 &
python LUNA16_extract_patches_subset_HDF5.py --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 1 &
python LUNA16_extract_patches_subset_HDF5.py --patch_size ${ARG1} --patch_depth ${ARG2} --subset 1 &

wait # Wait for the process above to stop

python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e ${ARG3} --patch_size ${ARG1} --patch_depth ${ARG2} --subset 0 &
python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e ${ARG3} --patch_size ${ARG1} --patch_depth ${ARG2} --subset 1 &

wait # Wait for the process above to stop

python LUNA16_inferenceTesting_ResNet_HDF.py -z 256 -i 0 -b gpu --patch_size ${ARG1} --patch_depth ${ARG2} --subset 0
python LUNA16_inferenceTesting_ResNet_HDF.py -z 256 -i 0 -b gpu --patch_size ${ARG1} --patch_depth ${ARG2} --subset 1

wait # Wait for the process above to stop

rm /mnt/data/medical/luna16/*.h5 

# Extract the ROI patches to HDF dataset files
python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 2 &
python LUNA16_extract_patches_subset_HDF5.py --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 2 & 
python LUNA16_extract_patches_subset_HDF5.py --patch_size ${ARG1} --patch_depth ${ARG2} --subset 2 &

python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 3 &
python LUNA16_extract_patches_subset_HDF5.py --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 3 &
python LUNA16_extract_patches_subset_HDF5.py --patch_size ${ARG1} --patch_depth ${ARG2} --subset 3 &

python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 4 &
python LUNA16_extract_patches_subset_HDF5.py --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 4 &
python LUNA16_extract_patches_subset_HDF5.py --patch_size ${ARG1} --patch_depth ${ARG2} --subset 4 &

wait # Wait for the process above to stop

python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e ${ARG3} --subset 2 &
python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e ${ARG3} --subset 3 &
python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e ${ARG3} --subset 4 &

wait # Wait for the process above to stop

python LUNA16_inferenceTesting_ResNet_HDF.py -z 256 -i 0 -b gpu --patch_size ${ARG1} --patch_depth ${ARG2} --subset 2
python LUNA16_inferenceTesting_ResNet_HDF.py -z 256 -i 0 -b gpu --patch_size ${ARG1} --patch_depth ${ARG2} --subset 3
python LUNA16_inferenceTesting_ResNet_HDF.py -z 256 -i 0 -b gpu --patch_size ${ARG1} --patch_depth ${ARG2} --subset 4

wait # Wait for the process above to stop

rm /mnt/data/medical/luna16/*.h5 

# Extract the ROI patches to HDF dataset files
python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 5 &
python LUNA16_extract_patches_subset_HDF5.py --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 5 & 
python LUNA16_extract_patches_subset_HDF5.py --patch_size ${ARG1} --patch_depth ${ARG2} --subset 5 &

python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 6 &
python LUNA16_extract_patches_subset_HDF5.py --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 6 &
python LUNA16_extract_patches_subset_HDF5.py --patch_size ${ARG1} --patch_depth ${ARG2} --subset 6 &

python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 7 &
python LUNA16_extract_patches_subset_HDF5.py --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 7 &
python LUNA16_extract_patches_subset_HDF5.py --patch_size ${ARG1} --patch_depth ${ARG2} --subset 7 &

wait # Wait for the process above to stop

python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e ${ARG3} --subset 5 &
python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e ${ARG3} --subset 6 &
python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e ${ARG3} --subset 7 &

wait # Wait for the process above to stop

python LUNA16_inferenceTesting_ResNet_HDF.py -z 256 -i 0 -b gpu --subset 5
python LUNA16_inferenceTesting_ResNet_HDF.py -z 256 -i 0 -b gpu --subset 6
python LUNA16_inferenceTesting_ResNet_HDF.py -z 256 -i 0 -b gpu --subset 7

wait # Wait for the process above to stop

rm /mnt/data/medical/luna16/*.h5 

# Extract the ROI patches to HDF dataset files
python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 8 &
python LUNA16_extract_patches_subset_HDF5.py --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 8 & 
python LUNA16_extract_patches_subset_HDF5.py --patch_size ${ARG1} --patch_depth ${ARG2} --subset 8 &

python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 9 &
python LUNA16_extract_patches_subset_HDF5.py --augment --patch_size ${ARG1} --patch_depth ${ARG2} --subset 9 &
python LUNA16_extract_patches_subset_HDF5.py --patch_size ${ARG1} --patch_depth ${ARG2} --subset 9 &

wait # Wait for the process above to stop

python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e ${ARG3} --subset 8 &
python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e ${ARG3} --subset 9 &

wait # Wait for the process above to stop

python LUNA16_inferenceTesting_ResNet_HDF.py -z 256 -i 0 -b gpu --subset 8
python LUNA16_inferenceTesting_ResNet_HDF.py -z 256 -i 0 -b gpu --subset 9

wait # Wait for the process above to stop

./process_predictions.sh 

cp predictions_ALL.csv ALL_predictions_patchsize${ARG1}_depth${ARG2}.csv

rm /mnt/data/medical/luna16/*.h5 

