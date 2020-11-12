

import random
import os
from PIL import Image, ImageChops, ImageMath
import numpy as np

F = os.path.dirname(os.path.abspath(__file__))

def change_background(img, mask, bg):
    '''
    Title: Real-Time Seamless Single Shot 6D Object Pose Prediction
    Author: Bugra Tekin
    Date: Oct 18, 2019
    Availability: https://github.com/microsoft/singleshotpose/blob/master/image.py
    '''
    
    ow, oh = img.size
    bg = bg.resize((ow, oh)).convert('RGB')
    
    imcs = list(img.split())
    bgcs = list(bg.split())
    maskcs = list(mask.split())
    fics = list(Image.new(img.mode, img.size).split())
    
    for c in range(len(imcs)):
        negmask = maskcs[c].point(lambda i: 1 - i / 255)
        posmask = maskcs[c].point(lambda i: i / 255)
        fics[c] = ImageMath.eval("a * c + b * d", a=imcs[c], b=bgcs[c], c=posmask, d=negmask).convert('L')
    out = Image.merge(img.mode, tuple(fics))

    return out



if __name__ == '__main__':
    
    background_dir = F + '\\..\\data\\background\\VOCdevkit\\VOC2012\\JPEGImages'
    backgroundImages = os.listdir(background_dir)
    random.shuffle(backgroundImages)
    
    images_dir = F + '\\..\\data\\car01\\APPLE_IPHONE_X\\SIMPLE\\frames'
    images     = os.listdir(images_dir)
    
    output_dir = F + '\\..\\OUT\\car01\\APPLE_IPHONE_X\\SIMPLE_AUGMENTED'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for imName in images:
        
        randomIDX = random.randint(0, len(backgroundImages))
        
        name = os.path.join(images_dir, imName)
    
        image = Image.open(name)
        mask = Image.open(name.replace('frames', 'binary_mask').replace('jpg', 'png')).convert('RGB')
        backgrounImage = Image.open(os.path.join(background_dir, backgroundImages[randomIDX]))

        output = change_background(image, mask, backgrounImage)
        output.save(os.path.join(output_dir, imName))