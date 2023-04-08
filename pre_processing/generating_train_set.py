from dataclasses import replace
from re import T
import numpy as np
import os
import cv2 as cv
import shutil
from alive_progress import alive_bar

png_dir = 'dataset/total/sub_png_reorder/'
mask_dir = 'dataset/total/sub_mask_reorder/'

png_img = os.listdir(png_dir)
mask_img = os.listdir(mask_dir)

count_val = 0
count_train = 0
count_test = 0
index_valtest_1 = np.random.choice(179, 40, replace = False)
index_valtest_0 = np.random.choice(len(png_img) - 179, 200, replace = False) + 179
index_val = np.r_[index_valtest_0[:100], index_valtest_1[:20]]
index_test = np.r_[index_valtest_0[100:], index_valtest_1[20:]]
# index_train = np.setdiff1d(np.arange(0, len(png_img)), index_val) 

shutil.rmtree('dataset/val_set/label/')
shutil.rmtree('dataset/train_set/label/')
shutil.rmtree('dataset/val_set/src/')
shutil.rmtree('dataset/train_set/src/')
shutil.rmtree('dataset/test_set/label/')
shutil.rmtree('dataset/test_set/src/')
os.mkdir('dataset/val_set/label/')
os.mkdir('dataset/train_set/label/')
os.mkdir('dataset/val_set/src/')
os.mkdir('dataset/train_set/src/')
os.mkdir('dataset/test_set/label/')
os.mkdir('dataset/test_set/src/')

# print(index_val_1)

with alive_bar(len(png_img)) as bar:
    for i in np.arange(0, len(png_img)):
        bar()
        imgname = str(i).zfill(6) + '.png'
        im = cv.imread(png_dir + imgname)
        mask = cv.imread(mask_dir + imgname)
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        if i in index_val:
            cv.imwrite('dataset/val_set/label/' + str(count_val).zfill(6) + '.png', mask)
            cv.imwrite('dataset/val_set/src/' + str(count_val).zfill(6) + '.png', im)
            count_val += 1
        elif i in index_test:
            cv.imwrite('dataset/test_set/label/' + str(count_test).zfill(6) + '.png', mask)
            cv.imwrite('dataset/test_set/src/' + str(count_test).zfill(6) + '.png', im)
            count_test += 1
        else:
            cv.imwrite('dataset/train_set/label/' + str(count_train).zfill(6) + '.png', mask)
            cv.imwrite('dataset/train_set/src/' + str(count_train).zfill(6) + '.png', im)
            count_train += 1

