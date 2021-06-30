import numpy as np
import tensorflow as tf
from keras.preprocessing import image

def data_augmentation(img, mask):
    if np.random.random() > 0.5:
        img = tf.image.flip_left_right(np.array(img))
        mask = tf.image.flip_left_right(np.array(mask))
    
#     if np.random.random() > 0.5:
#         img = tf.image.flip_up_down(np.array(img))
#         mask = tf.image.flip_up_down(np.array(mask))
    
    return np.array(img), np.array(mask)
    
def data_generator(img_datas, mask_datas, batch_size, img_size, train = False):
    
    
    while True:
        imgs = []
        masks = []
        
        rand_idx = np.random.choice(len(img_datas), batch_size)
        
        for i in rand_idx:
            img_data = img_datas[i]
            mask_data = mask_datas[i]
            
            img = image.load_img(img_data, target_size = (img_size, img_size, 3))
            img = np.array(img)
            mask = image.load_img(mask_data, target_size = (img_size, img_size), color_mode = 'grayscale')
            mask = np.array(mask).reshape(img_size, img_size, 1)
            
            if train:
                img, mask = data_augmentation(img, mask)
                
            imgs.append(img / 255.)
            masks.append(mask / 255.)
        yield np.array(imgs), np.array(masks)
