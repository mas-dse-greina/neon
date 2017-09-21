#!/usr/bin/env python

import matplotlib.pyplot as plt
import h5py
import numpy as np 

df = h5py.File('/mnt/data/medical/luna16/luna16_roi_subset1_augmented.h5')

print(df['input'].shape)

idx = df['input'].shape[0]

pos_idx = np.where(np.array(df['output']) == 1)[0]


lshape = df['input'].attrs['lshape']

j = 0

plots = 3
dim = np.round(np.true_divide(plots, 2)).astype(int)

np.random.shuffle(pos_idx)

plt.figure(figsize=(10,10))

def get_label(val):
	if val == 0:
		return 'Not cancer'
	else:
		return 'Cancer'

for i in pos_idx[:plots]:  #range(4,4+plots):
	a = df['input'][i,:]

	b = a.reshape(lshape)

	plt.subplot(dim, dim, j+1)
	plt.title(get_label(df['output'][i]))
	plt.imshow(b[int(lshape[0]//2),:,:], cmap='gray')
	j += 1

i = np.random.randint(0, idx)
plt.subplot(dim,dim,j+1)
plt.title(get_label(df['output'][i]))
a = df['input'][i,:]
b = a.reshape(lshape)
plt.imshow(b[int(lshape[0]//2),:,:], cmap='gray')

plt.show()

