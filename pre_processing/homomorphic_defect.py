import os
import cv2
from PIL import Image
import numpy as np

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def homomorphic_filter(src,d0=100,r1=0.4,rh=5,c=4,h=2.5,l=1):
    gray = src
    if len(src.shape)>2:
        gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows,cols = gray.shape
    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M,N = np.meshgrid(np.arange(-cols//2,cols//2),np.arange(-rows//2,rows//2))
    D = np.sqrt(M**2+N**2)
    Z = (rh-r1)*(1-np.exp(-c*(D**2/d0**2)))+r1
    dst_fftshift = Z*gray_fftshift
    dst_fftshift = (h-l)*dst_fftshift+l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst = np.uint8(np.clip(dst,0,255))
    return dst


# for name in os.listdir(imageDir):
#     img = Image.open(os.path.join(imageDir, name))
#     img = img.convert('L')
#     img = np.array(img)
#     #print(img,img.shape)
#     img_new = homomorphic_filter(img)
#     #print('new img shape is {}',format(img_new.shape))
#     #cv_show('1',img_new)
#     cv2.imwrite(os.path.join(saveDir, name),img_new)
def hisEqulColor(img):   
    img_eq = cv2.equalizeHist(img) #equalizeHist(in,out)  
    return img_eq  



defect_dir = 'dataset/defect/raw_jpg/'
nonedefect_dir = 'dataset/nonedefect/raw_jpg/'

if __name__ == "__main__":
    defect_img = os.listdir(defect_dir)
    nonedefect_img = os.listdir(nonedefect_dir)
    count = 0
    for imgname in defect_img:
        img = Image.open(defect_dir + imgname)
        img = img.convert('L')
        img = np.array(img)
        img = img[12:-12, 8:-8]
        #print(img,img.shape)
        img_new = homomorphic_filter(img)
        #print('new img shape is {}',format(img_new.shape))
        #cv_show('1',img_new)
        # cv2.imwrite(os.path.join(saveDir, name),img_new)
        # img = cv2.imread('Defect\CameraA-Snapshot-20220726-083504-357-16225119657.JPG', 0)
        # img_new = homomorphic_filter(img)

        # contrast = np.hstack((img, img_new))
        # img_eq = hisEqulColor(img_new)
        cv2.imwrite("dataset/total/png/" + str(count) + ".png", img_new)
        count += 1
    
        # cv2.imwrite('a.png', img_eq)
    # img_new = homomorphic_filter(img_new)
    # cv2.imwrite('b.png', img_new)
