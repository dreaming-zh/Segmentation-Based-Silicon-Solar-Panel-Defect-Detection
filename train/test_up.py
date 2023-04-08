

# test
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image


def array_img_reshape(array, target_shape):
    import PIL
    # PIL.Image.BILINEAR
    array = image.img_to_array(image.array_to_img(array).resize(target_shape, resample=PIL.Image.BILINEAR))
    return array

def preprocess_img(img_id,target_shape):
    '''
    加载图片，切分，归一化
    输出：子图集，最后一张子图的高度
    '''
    img = image.load_img(img_id)
    img = image.img_to_array(img)[...,:1]
    # img = img[12:-12, 8:-8]
    h, w, _ = np.shape(img)
    m = int(np.ceil(w/target_shape[1]))
    n = int(np.ceil(h/target_shape[0]))
    imgs = np.zeros((n*m, *target_shape, 1))
    for j in range(m):
        if j != m-1:
            for i in range(n):
                if i != n-1:
                    imgs[i + j*n,...] = img[target_shape[0]*i:target_shape[0]*(i+1), target_shape[0]*j:target_shape[0]*(j+1),...]
                else:
                    imgs[i + j*n,...] = array_img_reshape(img[target_shape[0]*i: , target_shape[0]*j:target_shape[0]*(j+1)], target_shape)
        else:
            for i in range(n):
                if i != n-1:
                    imgs[i + j*n,...] = array_img_reshape(img[target_shape[0]*i::target_shape[0]*(i+1) , target_shape[0]*j:], target_shape)
                else:
                    imgs[i + j*n,...] = array_img_reshape(img[target_shape[0]*i: , target_shape[0]*j:], target_shape)
    imgs = imgs/255.0
    return imgs, h-(n-1)*target_shape[0], w-(m-1)*target_shape[1]

def plot_result_P(sort_list_i, result, delta_h, delta_w, target_shape, class_i):
    img1  = image.load_img(sort_list_i[1]) # src
    img2  = image.load_img(sort_list_i[2]) # label
    imgs1 = np.zeros_like(img2)[...,:1]    # output1
    imgs2 = np.zeros_like(img2)[...,:1]    # output2
    
    h, w, _ = np.shape(img1)
    m = int(np.ceil(w/target_shape[1]))
    n = int(np.ceil(h/target_shape[0]))
    
    # for j in range(m):
    #     for i in range(n):
    #         plt.figure()
    #         plt.subplot(121)
    #         plt.imshow(image.array_to_img(image.img_to_array(img1)[target_shape[0]*i:target_shape[0]*(i+1), target_shape[0]*j:target_shape[0]*(j+1)]))
    #         plt.subplot(122)
    #         plt.imshow(result[0][i + j*n,...,0])
    #         plt.savefig("/root/zsh/result_debug/" + str(j) + '_' + str(i) + '.png')
    
    
    for j in range(m):
        if j != m-1:
            for i in range(n):
                fig_ij = result[0][i + j*n,...,:1]
                if i != n-1:
                    imgs1[target_shape[0]*i:target_shape[0]*(i+1), target_shape[0]*j:target_shape[0]*(j+1),...] = array_img_reshape(fig_ij, target_shape)
                else:
                    imgs1[target_shape[0]*i:, target_shape[0]*j:target_shape[0]*(j+1),...] = array_img_reshape(fig_ij, (target_shape[0], delta_h))
        else:
            for i in range(n):
                fig_ij = result[0][i + j*n,...,:1]
                if i != n-1:
                    imgs1[target_shape[0]*i:target_shape[0]*(i+1), target_shape[0]*j:,...] = array_img_reshape(fig_ij, (delta_w, target_shape[1]))
                else:
                    imgs1[target_shape[0]*i:, target_shape[0]*j:,...] = array_img_reshape(fig_ij, (delta_w, delta_h))
        
    for j in range(m):
        if j != m-1:
            for i in range(n):
                fig_ij = result[0][i + j*n,...,1:]
                if i != n-1:
                    imgs2[target_shape[0]*i:target_shape[0]*(i+1), target_shape[0]*j:target_shape[0]*(j+1),...] = array_img_reshape(fig_ij, target_shape)
                else:
                    imgs2[target_shape[0]*i:, target_shape[0]*j:target_shape[0]*(j+1),...] = array_img_reshape(fig_ij, (target_shape[0], delta_h))
        else:
            for i in range(n):
                fig_ij = result[0][i + j*n,...,1:]
                if i != n-1:
                    imgs2[target_shape[0]*i:target_shape[0]*(i+1), target_shape[0]*j:,...] = array_img_reshape(fig_ij, (delta_w, target_shape[1]))
                else:
                    imgs2[target_shape[0]*i:, target_shape[0]*j:,...] = array_img_reshape(fig_ij, (delta_w, delta_h))
    
    # for k in range(np.shape(result[0])[0]):
    #     fig_k = result[0][k,...,:1]
    #     if k == np.shape(result[0])[0]-1:
    #         imgs1[target_size[0]*k:,...] = array_img_reshape(fig_k, (target_size[0], delta_h))
    #     else:
    #         imgs1[target_size[0]*k:target_size[0]*(k+1),...] = array_img_reshape(fig_k, target_size)
    # for j in range(np.shape(result[0])[0]):
    #     fig_j = result[0][j,...,1:]
    #     if j == np.shape(result[0])[0]-1:
    #         imgs2[target_size[0]*j:,...] = array_img_reshape(fig_j, (target_size[0], delta_h))
    #     else:
    #         imgs2[target_size[0]*j:target_size[0]*(j+1),...] = array_img_reshape(fig_j, target_size)
            
            
            
    imgs2 = np.concatenate((imgs1,imgs2),axis=2)
    imgs2 = np.reshape(imgs2,(1,-1,2))
    imgs2 = np.argmax(imgs2,axis=2).astype(np.int8)
    imgs2 = np.reshape(imgs2,np.shape(imgs1))
    
    if class_i != sort_list_i[0]: 
        type = 'FP'
    else:
        type = 'TP'
    plt.figure(figsize=(10,10))
    plt.title(type)
    plt.subplot(221)
    plt.imshow(img1)
    plt.title('src')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img2)
    plt.title('label')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(255-imgs1[...,0],cmap='gray')
    plt.title('output')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(imgs2[...,0],cmap='gray')
    plt.title('result')
    plt.axis('off')
    plt.savefig("/root/zsh/result_huge/" + type + '_' + sort_list_i[1][-10:])
    #plt.show()
    return None

def plot_result_N(sort_list_i, result, delta_h, delta_w, target_shape, class_i):
    img1  = image.load_img(sort_list_i[1]) # src
    img2  = image.load_img(sort_list_i[2]) # label
    imgs1 = np.zeros_like(img2)[...,:1]    # output1
    imgs2 = np.zeros_like(img2)[...,:1]    # output2
    
    h, w, _ = np.shape(img1)
    m = int(np.ceil(w/target_shape[1]))
    n = int(np.ceil(h/target_shape[0]))
    
    
    
    for j in range(m):
        if j != m-1:
            for i in range(n):
                fig_ij = result[0][i + j*n,...,:1]
                if i != n-1:
                    imgs1[target_shape[0]*i:target_shape[0]*(i+1), target_shape[0]*j:target_shape[0]*(j+1),...] = array_img_reshape(fig_ij, target_shape)
                else:
                    imgs1[target_shape[0]*i:, target_shape[0]*j:target_shape[0]*(j+1),...] = array_img_reshape(fig_ij, (target_shape[0], delta_h))
        else:
            for i in range(n):
                fig_ij = result[0][i + j*n,...,:1]
                if i != n-1:
                    imgs1[target_shape[0]*i:target_shape[0]*(i+1), target_shape[0]*j:,...] = array_img_reshape(fig_ij, (delta_w, target_shape[1]))
                else:
                    imgs1[target_shape[0]*i:, target_shape[0]*j:,...] = array_img_reshape(fig_ij, (delta_w, delta_h))
        
    for j in range(m):
        if j != m-1:
            for i in range(n):
                fig_ij = result[0][i + j*n,...,1:]
                if i != n-1:
                    imgs2[target_shape[0]*i:target_shape[0]*(i+1), target_shape[0]*j:target_shape[0]*(j+1),...] = array_img_reshape(fig_ij, target_shape)
                else:
                    imgs2[target_shape[0]*i:, target_shape[0]*j:target_shape[0]*(j+1),...] = array_img_reshape(fig_ij, (target_shape[0], delta_h))
        else:
            for i in range(n):
                fig_ij = result[0][i + j*n,...,1:]
                if i != n-1:
                    imgs2[target_shape[0]*i:target_shape[0]*(i+1), target_shape[0]*j:,...] = array_img_reshape(fig_ij, (delta_w, target_shape[1]))
                else:
                    imgs2[target_shape[0]*i:, target_shape[0]*j:,...] = array_img_reshape(fig_ij, (delta_w, delta_h))
    
    # for k in range(np.shape(result[0])[0]):
    #     fig_k = result[0][k,...,:1]
    #     if k == np.shape(result[0])[0]-1:
    #         imgs1[target_size[0]*k:,...] = array_img_reshape(fig_k, (target_size[0], delta_h))
    #     else:
    #         imgs1[target_size[0]*k:target_size[0]*(k+1),...] = array_img_reshape(fig_k, target_size)
    # for j in range(np.shape(result[0])[0]):
    #     fig_j = result[0][j,...,1:]
    #     if j == np.shape(result[0])[0]-1:
    #         imgs2[target_size[0]*j:,...] = array_img_reshape(fig_j, (target_size[0], delta_h))
    #     else:
    #         imgs2[target_size[0]*j:target_size[0]*(j+1),...] = array_img_reshape(fig_j, target_size)
            
            
            
    imgs2 = np.concatenate((imgs1,imgs2),axis=2)
    imgs2 = np.reshape(imgs2,(1,-1,2))
    imgs2 = np.argmax(imgs2,axis=2).astype(np.int8)
    imgs2 = np.reshape(imgs2,np.shape(imgs1))
    
    if class_i != sort_list_i[0]: 
        type = 'FN'
    else:
        type = 'TN'
    plt.figure(figsize=(10,10))
    plt.title(type)
    plt.subplot(221)
    plt.imshow(img1)
    plt.title('src')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img2)
    plt.title('label')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(255-imgs1[...,0],cmap='gray')
    plt.title('output')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(imgs2[...,0],cmap='gray')
    plt.title('result')
    plt.axis('off')
    plt.savefig("/root/zsh/result_huge/" + type + '_' + sort_list_i[1][-10:])
    #plt.show()
    return None


def load_model_from_json(model_path, model_name):
    with open(model_path+model_name+'.json', 'r') as file:
        model_json = file.read()
    model = model_from_json(model_json)
    model.load_weights(model_path+model_name+'_acc_weights.h5')
    return model

if __name__ == '__main__':
    # 加载测试数据
    import data_manager
    #test_set_path  = './KolektorSDD/test_set'
    test_set_path  = '/root/zsh/dataset/total'
    sort_list = data_manager.data_manager_huge(test_set_path)
    # 模型的输入形状
    target_size = (608, 608)
    output_size = (608, 608)
    n_label = 2
    # 加载模型 / 权重
    model_path = '/root/zsh/model/'
    json_name = '20221215-204620_v2'
    model_loadname = '20221215-204620_v2'

    try:
        '''
        两者选其一
        1. 直接加载模型
        2. 加载json文件和weights权重
        '''
        # model = load_model(model_path+model_name+'_acc_weights.h5')     # 1.
        model = load_model_from_json(model_path, json_name) # 2.
        model.load_weights(model_path + model_loadname + '_acc_weights.h5')
        print('Model weights loaded!')
    except:
        raise FileNotFoundError('Model not found!')
    
    model.summary()
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(len(sort_list)):
        print('开始处理第',str(i+1),'张图片...')
        t1 = time.time()
        img_set, delta_h, delta_w = preprocess_img(sort_list[i][1],target_size)
        result = model.predict(img_set, verbose=0)
        # 判断所有子图中是否包含class为 1 的输出
        class_i = 0
        for j in range(np.shape(result[0])[0]):
            class_i = np.max((class_i, int(list(np.rint(result[1][j])).index(1)), 0))
        t2 = time.time()
        if class_i != sort_list[i][0]:
            print('第{}例分类错误！耗时：{:.4f}s'.format(i+1, t2-t1))
            # 画出分类错误的实例
            # plot_result(sort_list[i], result, delta_h, target_size)
        else:
            print('第{}例分类正确！耗时：{:.4f}s'.format(i+1, t2-t1))
            # 画出分类正确的正例
            # plot_result(sort_list[i], result, delta_h, target_size)
        if class_i == 1:
            plot_result_P(sort_list[i], result, delta_h, delta_w, target_size, class_i)
        else:
            plot_result_N(sort_list[i], result, delta_h, delta_w, target_size, class_i)
        # 统计 TP, FP, FN, TN
        if class_i == 1:                 # P
            if sort_list[i][0] == 1:     # T
                TP += 1
            elif sort_list[i][0] == 0:   # F
                FP += 1
        elif class_i == 0:               # N
            if sort_list[i][0] == 1:     # F
                FN += 1
            elif sort_list[i][0] == 0:   # T
                TN += 1
    print('TP:{}  FP:{}  FN:{}  TN:{}'.format(TP, FP, FN, TN))
    print('准确率：%.3f%%'%((TP+TN)/(TP+TN+FP+FN)*100))
    print('查准率：%.3f%%'%(TP/(TP+FP)*100))
    print('查全率：%.3f%%'%(TP/(TP+FN)*100))
    