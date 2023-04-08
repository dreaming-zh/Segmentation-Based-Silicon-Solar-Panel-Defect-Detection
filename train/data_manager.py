
# 数据管理

import os
import numpy as np
from tensorflow.keras.preprocessing import image
import shutil
from alive_progress import alive_bar

def data_manager_huge(data_path):
    '''
    构建数据管理器
    返回：分类、数据、标签
    '''
    fold_list = os.listdir(data_path)
    data_list = []
    label_list= []
    class_list= []
    sort_list = []
    for i in fold_list:
        if 'debug' in i:
            continue
        file_list = os.listdir(data_path + '/' + i)
        for j in file_list:
            if 'mask' in i:
                label_list.append(data_path + '/' + i + '/' + j)
            elif 'png' in i:
                data_list.append(data_path + '/' + i + '/' + j)
    for i in label_list:
        img = image.load_img(i)
        img = image.img_to_array(img)
        if np.count_nonzero(img)==0:
            class_list.append(0)
        else:
            class_list.append(1)
    for i in range(len(class_list)):
        sort_list.append([class_list[i],data_list[i],label_list[i]])
    sort_list.sort()
    return sort_list

def data_manager(data_path):
    '''
    构建数据管理器
    返回：分类、数据、标签
    '''
    fold_list = os.listdir(data_path)
    data_list = []
    label_list= []
    class_list= []
    sort_list = []
    for i in fold_list:
        if not 'aug' in i:
            continue
        file_list = os.listdir(data_path + '/' + i)
        for j in file_list:
            if 'label' in i:
                label_list.append(data_path + '/' + i + '/' + j)
            elif 'src' in i:
                data_list.append(data_path + '/' + i + '/' + j)
    for i in label_list:
        img = image.load_img(i)
        img = image.img_to_array(img)
        if np.count_nonzero(img)==0:
            class_list.append(0)
        else:
            class_list.append(1)
    for i in range(len(class_list)):
        sort_list.append([class_list[i],data_list[i],label_list[i]])
    sort_list.sort()
    return sort_list


def depart(sort_list):
    for i in range(len(sort_list)):
        class_i = sort_list[i][0]
        if class_i == 1:
            return i
def augmentation_img(img1,img2):
    '''
    图像增强（旋转/镜像翻转/错位缩放/亮度调整/噪声）
    '''
    aug = np.random.randint(2,size=4)
    img_a1, img_a2 = np.copy(img1) , np.copy(img2)
    
    if aug[0]:# 旋转
        angle  = np.random.choice((1,2,-1,-2))
        img_a1 = np.rot90(img_a1,angle)
        img_a2 = np.rot90(img_a2,angle)
    if aug[1]:# 对称翻转 
        x_or_y = np.random.choice((0,1))
        img_a1 = np.flip(img_a1,x_or_y)
        img_a2 = np.flip(img_a2,x_or_y)
    # if aug[4]:# 缩放
    #     zoom = np.random.uniform(0.6,1.0)
    #     img_a1 = image.random_zoom(img_a1,(zoom,zoom),0,1,2) 
    #     img_a2 = image.random_zoom(img_a2,(zoom,zoom),0,1,2) 
    if aug[3]:# 亮度
        img_a1 = image.random_brightness(img_a1, (0.5,1.5))
    if aug[2]:# 噪声
        img_a1 += np.random.uniform(0,5,(np.shape(img_a1)))
        img_a1 = np.clip(img_a1,0,255)
    # 转成 0-255 整数
    img_a1 = np.array(img_a1,'uint8')
    img_a2 = np.array(img_a2,'uint8')
    return img_a1, img_a2

def data_aug_generator(mode,data_list,count_n,savepath,aug,shape=(608,608,3)):
    '''
    通过数据增强制造更多样本
    '''
    with alive_bar(count_n) as bar:
        for count in range(count_n):
            bar()
            i = np.random.randint(0,len(data_list)-1)
            src = image.load_img(data_list[i][1])
            lab = image.load_img(data_list[i][2])
            src = image.img_to_array(src)
            lab = image.img_to_array(lab)
            h,w,_ = src.shape
            point = 0
            src = src[point:point+shape[0],:,2:]
            lab = lab[point:point+shape[0],:,2:]
            if mode:# 1保存含有缺陷的样本
                if aug:
                    src, lab = augmentation_img(src, lab)
                if np.count_nonzero(lab)!=0:
                    count += 1
                    src = image.array_to_img(src)
                    lab = image.array_to_img(lab)
                    src.save(savepath+'/src_aug/1_'+str(count).zfill(6)+'.png')
                    lab.save(savepath+'/label_aug/1_'+str(count).zfill(6)+'.png')
            else:# 0保存不含缺陷的样本
                if aug:
                    src, lab = augmentation_img(src, lab)
                if np.count_nonzero(lab)==0:
                    count += 1
                    src = image.array_to_img(src)
                    lab = image.array_to_img(lab)
                    src.save(savepath+'/src_aug/0_'+str(count).zfill(6)+'.png')
                    lab.save(savepath+'/label_aug/0_'+str(count).zfill(6)+'.png')
    return None

if __name__ == '__main__':
    train_set_load_path = '/root/zsh/dataset/train_set'
    val_set_save_path   = '/root/zsh/dataset/val_set'
    train_set_save_path = '/root/zsh/dataset/train_set'
    val_set_load_path   = '/root/zsh/dataset/val_set'
    test_set_save_path = '/root/zsh/dataset/test_set'
    test_set_load_path   = '/root/zsh/dataset/test_set'

    # shutil.rmtree(train_set_save_path + '/src_aug')
    os.mkdir(train_set_save_path + '/src_aug')
    # shutil.rmtree(train_set_save_path + '/label_aug')
    os.mkdir(train_set_save_path + '/label_aug')
    # shutil.rmtree(val_set_save_path + '/src_aug')
    os.mkdir(val_set_save_path + '/src_aug')
    # shutil.rmtree(val_set_save_path + '/label_aug')
    os.mkdir(val_set_save_path + '/label_aug')
    # shutil.rmtree(test_set_save_path + '/src_aug')
    os.mkdir(test_set_save_path + '/src_aug')
    # shutil.rmtree(test_set_save_path + '/label_aug')
    os.mkdir(test_set_save_path + '/label_aug')
    # 训练数据集
    sort_list = data_manager(train_set_load_path)
    print('-------------------generating train set-----------------')
    data_aug_generator(0,sort_list[:depart(sort_list)],2000,train_set_save_path,1)
    data_aug_generator(1,sort_list[depart(sort_list):],2000,train_set_save_path,1)
    print('-------------------DONE train set-----------------!')

    # 测试数据集
    sort_list = data_manager(val_set_load_path)
    print('-------------------generating val set-----------------')
    data_aug_generator(0,sort_list[:depart(sort_list)],500,val_set_save_path,1)
    data_aug_generator(1,sort_list[depart(sort_list):],500,val_set_save_path,1)
    print('-------------------DONE val set-----------------')

    sort_list = data_manager(test_set_load_path)
    print('-------------------generating test set-----------------')
    data_aug_generator(0,sort_list[:depart(sort_list)],500,test_set_save_path,1)
    data_aug_generator(1,sort_list[depart(sort_list):],500,test_set_save_path,1)
    print('-------------------DONE test set-----------------')

