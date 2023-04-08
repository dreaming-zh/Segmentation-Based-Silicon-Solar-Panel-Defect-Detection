import numpy as np
import os

import numpy as np
import cv2 as cv
import os
import shutil

png_dir = 'dataset/total/sub_png/'
mask_dir = 'dataset/total/sub_mask/'

png_img = os.listdir(png_dir)
mask_img = os.listdir(mask_dir)

shutil.rmtree('dataset/total/sub_mask_reorder/')
os.mkdir('dataset/total/sub_mask_reorder/')
shutil.rmtree('dataset/total/sub_png_reorder/')
os.mkdir('dataset/total/sub_png_reorder/')


count = 0
for imgname in mask_img:
    im = cv.imread(png_dir + imgname)
    mask = cv.imread(mask_dir + imgname)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    if len(np.where(mask[20:-20, 20:-20] > 254)[0]) > 700:
        cv.imwrite('dataset/total/sub_mask_reorder/' + str(count).zfill(6) + '.png', mask)
        cv.imwrite('dataset/total/sub_png_reorder/' + str(count).zfill(6) + '.png', im)
        count += 1
        
for imgname in mask_img:
    im = cv.imread(png_dir + imgname)
    mask = cv.imread(mask_dir + imgname)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    if len(np.where(mask[20:-20, 20:-20] > 254)[0]) <= 700:
        mask = np.zeros(np.shape(mask))
        cv.imwrite('dataset/total/sub_mask_reorder/' + str(count).zfill(6) + '.png', mask)
        cv.imwrite('dataset/total/sub_png_reorder/' + str(count).zfill(6) + '.png', im)
        count += 1