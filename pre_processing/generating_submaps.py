import numpy as np
import cv2 as cv
import os

png_dir = 'dataset/defect/raw_jpg/'
mask_dir = 'dataset/total/mask/'

png_img = os.listdir(png_dir)
mask_img = os.listdir(mask_dir)
count = 0
step = 608
for imgname in png_img:
    im = cv.imread(png_dir + imgname)
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im = im[12:-12, 8:-8]
    for i in range(int(np.shape(im)[0]/step)):
        for j in range(int(np.shape(im)[1]/step)):
            im_sub = im[i*step:(i + 1)*step, j*step:(j + 1)*step]
            cv.imwrite('dataset/total/sub_png_raw/' + str(count).zfill(6) + '.png', im_sub)
            count += 1
    
png_dir = 'dataset/nonedefect/raw_jpg/'
png_img = os.listdir(png_dir)
for imgname in png_img:
    im = cv.imread(png_dir + imgname)
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im = im[12:-12, 8:-8]
    for i in range(int(np.shape(im)[0]/step)):
        for j in range(int(np.shape(im)[1]/step)):
            im_sub = im[i*step:(i + 1)*step, j*step:(j + 1)*step]
            cv.imwrite('dataset/total/sub_png_raw/' + str(count).zfill(6) + '.png', im_sub)
            count += 1
count = 0        
for imgname in mask_img:
    im = cv.imread(mask_dir + imgname)
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    for i in range(int(np.shape(im)[0]/step)):
        for j in range(int(np.shape(im)[1]/step)):
            im_sub = im[i*step:(i + 1)*step, j*step:(j + 1)*step]
            cv.imwrite('dataset/total/sub_mask_raw/' + str(count).zfill(6) + '.png', im_sub)
            count += 1