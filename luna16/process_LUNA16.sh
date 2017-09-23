#!/bin/bash

# Extract the ROI patches to HDF dataset files
python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --subset 0 &
python LUNA16_extract_patches_subset_HDF5.py --augment --subset 0 &
python LUNA16_extract_patches_subset_HDF5.py --subset 0 &

python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --subset 1 &
python LUNA16_extract_patches_subset_HDF5.py --augment --subset 1 &
python LUNA16_extract_patches_subset_HDF5.py --subset 1 &

wait # Wait for the process above to stop

python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e 40 --subset 0 &
python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e 40 --subset 1 &

wait # Wait for the process above to stop

python LUNA16_inferenceTesting_ResNet_HDF.py -z 1024 -i 0 -b gpu --subset 0
python LUNA16_inferenceTesting_ResNet_HDF.py -z 1024 -i 0 -b gpu --subset 1

wait # Wait for the process above to stop

rm /mnt/data/medical/luna16/*.h5 

# Extract the ROI patches to HDF dataset files
python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --subset 2 &
python LUNA16_extract_patches_subset_HDF5.py --augment --subset 2 & 
python LUNA16_extract_patches_subset_HDF5.py --subset 2 &

python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --subset 3 &
python LUNA16_extract_patches_subset_HDF5.py --augment --subset 3 &
python LUNA16_extract_patches_subset_HDF5.py --subset 3 &

python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --subset 4 &
python LUNA16_extract_patches_subset_HDF5.py --augment --subset 4 &
python LUNA16_extract_patches_subset_HDF5.py --subset 4 &

wait # Wait for the process above to stop

python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e 40 --subset 2 &
python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e 40 --subset 3 &
python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e 40 --subset 4 &

wait # Wait for the process above to stop

python LUNA16_inferenceTesting_ResNet_HDF.py -z 1024 -i 0 -b gpu --subset 2
python LUNA16_inferenceTesting_ResNet_HDF.py -z 1024 -i 0 -b gpu --subset 3
python LUNA16_inferenceTesting_ResNet_HDF.py -z 1024 -i 0 -b gpu --subset 4

wait # Wait for the process above to stop

rm /mnt/data/medical/luna16/*.h5 

# Extract the ROI patches to HDF dataset files
python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --subset 5 &
python LUNA16_extract_patches_subset_HDF5.py --augment --subset 5 & 
python LUNA16_extract_patches_subset_HDF5.py --subset 5 &

python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --subset 6 &
python LUNA16_extract_patches_subset_HDF5.py --augment --subset 6 &
python LUNA16_extract_patches_subset_HDF5.py --subset 6 &

python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --subset 7 &
python LUNA16_extract_patches_subset_HDF5.py --augment --subset 7 &
python LUNA16_extract_patches_subset_HDF5.py --subset 7 &

wait # Wait for the process above to stop

python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e 40 --subset 5 &
python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e 40 --subset 6 &
python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e 40 --subset 7 &

wait # Wait for the process above to stop

python LUNA16_inferenceTesting_ResNet_HDF.py -z 1024 -i 0 -b gpu --subset 5
python LUNA16_inferenceTesting_ResNet_HDF.py -z 1024 -i 0 -b gpu --subset 6
python LUNA16_inferenceTesting_ResNet_HDF.py -z 1024 -i 0 -b gpu --subset 7

wait # Wait for the process above to stop

rm /mnt/data/medical/luna16/*.h5 

# Extract the ROI patches to HDF dataset files
python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --subset 8 &
python LUNA16_extract_patches_subset_HDF5.py --augment --subset 8 & 
python LUNA16_extract_patches_subset_HDF5.py --subset 8 &

python LUNA16_extract_patches_subset_HDF5.py --exclude --augment --subset 9 &
python LUNA16_extract_patches_subset_HDF5.py --augment --subset 9 &
python LUNA16_extract_patches_subset_HDF5.py --subset 9 &

wait # Wait for the process above to stop

python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e 40 --subset 8 &
python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 --depth 50 -e 40 --subset 9 &

wait # Wait for the process above to stop

python LUNA16_inferenceTesting_ResNet_HDF.py -z 1024 -i 0 -b gpu --subset 8
python LUNA16_inferenceTesting_ResNet_HDF.py -z 1024 -i 0 -b gpu --subset 9

wait # Wait for the process above to stop

rm /mnt/data/medical/luna16/*.h5 

