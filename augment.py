# custom_augmentation.py
import numpy as np
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import matplotlib.pyplot as plt

I_0 = 256
# stain_mat = np.array([
#     [ 0.53420104, -0.8391861 , -0.08528147],
#     [ 0.79607249,  0.46660234,  0.38413706],
#     [ 0.28257426,  0.27306047, -0.91785066]
# ])

def rgb_to_sda(img) :        
    im_rgb = img.astype(float) + 1
    im_sda = -np.log(im_rgb/(1.*I_0)) * 255/np.log(I_0)
    return im_sda


def sda_to_rgb(img):
    im_rgb = I_0 ** (1 - img / 255.) - 1
    return im_rgb

def get_svd(image) :
    img = image.copy()
#     img = img.reshape(-1, 3) - img.reshape(-1, 3).mean(axis=0)
    img = img.reshape(-1, 3)
    U, sigma, V = np.linalg.svd(img.T, full_matrices=False)
    return U, sigma

def get_svd_with_mask(image, mask_out) :
    img = image.copy()
    
    if mask_out is not None:
        keep_mask = np.equal(mask_out[..., None], False)
        keep_mask = np.tile(keep_mask, (1, 1, 3))
        keep_mask = keep_mask.reshape(-1, keep_mask.shape[-1]).T
        img = img.reshape(-1, 3).T
        img = img[:, keep_mask.all(axis=0)]    
    
    img = img.reshape(-1, 3) - img.reshape(-1, 3).mean(axis=0)
    U, sigma, V = np.linalg.svd(img.T, full_matrices=False)
    return U, sigma

def get_pca(image) :
    img = image.copy()
    img_flat = img.reshape(-1,3)
    img_flat = img_flat - np.mean(img_flat, axis=0)
    rgb_covariance = np.matmul(img_flat.T, img_flat)/(img_flat.shape[0] -1)
    eigenvalue, eigenmat = np.linalg.eig(rgb_covariance)
    return eigenmat

def HEColor_augment(img, stain_mat, inv_mat, sigma1=0.1, sigma2=1):
    # conversion to Stain & eosin domain
    # mask_out = img.mean(axis=2) >= 250
    sda_img = rgb_to_sda(255. - img.astype(float))   
    
    ratio = 1.
    if stain_mat is None :
        stain_mat, sigma = get_svd(sda_img)
        x,y,z = np.sqrt(sigma)
        ratio *= 3 / (x/z)
        inv_mat = stain_mat.T
    
    conv_img = np.matmul(sda_img, inv_mat.T)

    # remove scale and add more bias for main axis, axis ratio is about 20 : 2 : 1
    alpha1 = np.random.uniform(1-0.1*ratio, 1+0.1*ratio)
    for i in range(conv_img.shape[-1]):
        if i == 0 :
            alpha = alpha1 * np.random.uniform(1-sigma1*ratio, 1+sigma1*ratio)
            beta = np.random.uniform(-sigma2, sigma2)
        elif i == 1 :
            alpha = alpha1 * np.random.uniform(1-sigma1*ratio, 1+sigma1*ratio)
            beta = np.random.uniform(-sigma2, sigma2)
        else :
            alpha = alpha1 * np.random.uniform(1-sigma1*ratio, 1+sigma1*ratio)
            beta = np.random.uniform(-sigma2, sigma2)         
        conv_img[:,:,i] *= alpha
        conv_img[:,:,i] += beta
        
    aug_img = np.matmul(conv_img, stain_mat.T)
    aug_img = sda_to_rgb(aug_img)
    
    return (255 - np.clip(aug_img, 0, 255)).astype(np.uint8)


class HEColorAugment(ImageOnlyTransform) :    
    def __init__(self, sigma1=.5, sigma2=2., mat=None, always_apply=False, p=1.):
        super(HEColorAugment, self).__init__(always_apply, p)
        self.sigma1= sigma1
        self.sigma2= sigma2
        if sigma1 >= 1. :
            print('warning, sigma1 is more than 1.')
        
        # Define stain norm matrix
        if mat is None:
            # get stain mat per image
            self.stain_mat = None
            self.inverse_mat = None
            
        else:
            self.stain_mat = mat
            self.inverse_mat = np.linalg.inv(self.stain_mat)        
        
    def apply(self, img, **params):
        return HEColor_augment(img, self.stain_mat, self.inverse_mat, self.sigma1, self.sigma2, )
    
transform = A.Compose([
    HEColorAugment(sigma1=.4, sigma2=1., mat=None, p=1.),
#    A.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=1.)
])    
    
def plot_aug(img) :
    aug_list = []
    for i in range(7) :
        transformed = transform(image=img )
        aug_img = transformed['image'].astype(np.uint16)
        aug_list.append(aug_img)

    fig = plt.figure(figsize=(18, 4))
    fig.tight_layout()

    rows = 1
    cols = 8

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(img)
    ax1.set_title('org')
    ax1.axis("off")

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(aug_list[0])
    ax2.set_title('img2')
    ax2.axis("off")

    ax3 = fig.add_subplot(rows, cols, 3)
    ax3.imshow(aug_list[1])
    ax3.set_title('img3')
    ax3.axis("off")

    ax4 = fig.add_subplot(rows, cols, 4)
    ax4.imshow(aug_list[2])
    ax4.set_title('img4')
    ax4.axis("off")

    ax5 = fig.add_subplot(rows, cols, 5)
    ax5.imshow(aug_list[3])
    ax5.set_title('img5')
    ax5.axis("off")

    ax6 = fig.add_subplot(rows, cols, 6)
    ax6.imshow(aug_list[4])
    ax6.set_title('img6')
    ax6.axis("off")

    ax7 = fig.add_subplot(rows, cols, 7)
    ax7.imshow(aug_list[5])
    ax7.set_title('img7')
    ax7.axis("off")

    ax8 = fig.add_subplot(rows, cols, 8)
    ax8.imshow(aug_list[6])
    ax8.set_title('img8')
    ax8.axis("off")

    plt.show()    
    
