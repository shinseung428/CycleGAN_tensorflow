import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.io as sio
import scipy.ndimage

def save_images(orig, refined, cycled, path):

    new_img = np.concatenate((orig,refined), axis=2)
    new_img = np.concatenate((new_img,cycled), axis=2)
    new_img = np.reshape(new_img, (new_img.shape[1], new_img.shape[2], new_img.shape[3]))

    new_img_ = np.zeros((new_img.shape[0], new_img.shape[1], 3), dtype=np.float)
    new_img_[:,:,0] = new_img_[:,:,1] = new_img_[:,:,2] = new_img[:,:,0]

    scipy.misc.imsave(path , new_img_)

def save_image(image, path):

    new_img = np.reshape(image, (image.shape[1], image.shape[2], image.shape[3]))
    new_img_ = np.zeros((new_img.shape[0], new_img.shape[1], 3), dtype=np.float)
    new_img_[:,:,0] = new_img_[:,:,1] = new_img_[:,:,2] = new_img[:,:,0]

    scipy.misc.imsave(path , new_img_)

def imread(path, is_gray=False):
    if(is_gray):
        return scipy.misc.imread(path).astype(np.float)
    else:
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

def transform(image, input_h, input_w,
              resize_h, resize_w):
    image_ = scipy.misc.imresize(image, [resize_h, resize_w])
    return np.array(image_)/127.5 - 1.

def load_image(path,
               input_height, input_width,
               resize_height=128, resize_width=128,
               is_gray=False):
    image = imread(path, is_gray)
    tmp = transform(image, input_height, input_width, resize_height, resize_width)

    return transform(image, input_height, input_width, resize_height, resize_width)

def read_joints(path):
    joint_file = open(path)
    joint_arr = joint_file.read().split('\r\n')
    return joint_arr[0:len(joint_arr)-1]