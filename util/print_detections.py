

import os
import cv2
import numpy
import random


F = os.path.dirname(os.path.abspath(__file__))


def draw_corners(image, points, color=(0, 255, 0), front= (255, 255, 0), 
                 back= (0, 255, 255), size=5, text=False, use_same_color=False):
    image_cp = image.copy()
    
    p = []

    if use_same_color:
        front, back = color, color

    for i, point in enumerate(points):
        if len(point) == 1:
            x, y = int(point[0][0]), int(point[0][1])
            if i != 0 or len(points) == 8:
                p.append([x,y])
        else:
            x, y = int(point[0]), int(point[1])
            if i != 0 or len(points) == 8:
                p.append([x,y])

        if i == 0:
            current_color = color
        elif i < 5:
            current_color = back
        else:
            current_color = front
        cv2.circle(image_cp, (x, y), size, current_color, -1)
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_cp , str(i), (x, y - 5), font, 1, color, 3, cv2.LINE_AA)

    cv2.line(image_cp, (p[0][0], p[0][1]), (p[1][0], p[1][1]), back, 4)
    cv2.line(image_cp, (p[0][0], p[0][1]), (p[2][0], p[2][1]), back, 4)
    cv2.line(image_cp, (p[3][0], p[3][1]), (p[1][0], p[1][1]), back, 4)
    cv2.line(image_cp, (p[3][0], p[3][1]), (p[2][0], p[2][1]), back, 4)

    cv2.line(image_cp, (p[0][0], p[0][1]), (p[4][0], p[4][1]), color, 4)
    cv2.line(image_cp, (p[5][0], p[5][1]), (p[1][0], p[1][1]), color, 4)
    cv2.line(image_cp, (p[6][0], p[6][1]), (p[2][0], p[2][1]), color, 4)
    cv2.line(image_cp, (p[7][0], p[7][1]), (p[3][0], p[3][1]), color, 4)

    cv2.line(image_cp, (p[5][0], p[5][1]), (p[4][0], p[4][1]), front, 4)
    cv2.line(image_cp, (p[6][0], p[6][1]), (p[4][0], p[4][1]), front, 4)
    cv2.line(image_cp, (p[7][0], p[7][1]), (p[5][0], p[5][1]), front, 4)
    cv2.line(image_cp, (p[7][0], p[7][1]), (p[6][0], p[6][1]), front, 4)

    return image_cp


def execute_():
    # Annotations images and labels
    IMAGES_FOLDER     = F + '\\..\\data\\car01\\APPLE_IPHONE_X\\SIMPLE\\frames'
    
    PR_CORNERS_FOLDER = F + '\\..\\data\\car01\\APPLE_IPHONE_X\\SIMPLE\\corners'
    GT_CORNERS_FOLDER = F + '\\..\\data\\car01\\APPLE_IPHONE_X\\SIMPLE\\corners'
    
    OUTPUT_FOLDER = F + '\\..\\OUT\\detections\\car01\\IX'
    
    ADDGT = True

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)


    PREDICTIONS_IMG_SHAPE = (680, 680)    
    TARGET_IMAGE_SIZE     = (1920, 1080)
    
    OUTPUT_VIDEO_FILE =  OUTPUT_FOLDER + '\\car01_ix_simple_detection.mp4'
    
    if not os.path.exists(os.path.dirname(OUTPUT_VIDEO_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_VIDEO_FILE))
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, 30.0, TARGET_IMAGE_SIZE)

    IMAGES_NAMES = os.listdir(IMAGES_FOLDER)

    #
    IMAGES_NAMES = [int(im.replace('.jpg', '')) for im in IMAGES_NAMES]
    IMAGES_NAMES.sort()
    IMAGES_NAMES = [str(imid) + '.jpg' for imid in IMAGES_NAMES]
    
    for idx, image_filename in enumerate(IMAGES_NAMES):
        
        image = cv2.imread(os.path.join(IMAGES_FOLDER, image_filename))
        H, W, C = image.shape
        
        image = cv2.resize(image, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        H, W = TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0]
        
        annotation = numpy.loadtxt(os.path.join(PR_CORNERS_FOLDER, '{}.txt'.format(idx))).flatten()
        
        corners_values = numpy.reshape(annotation, (9, 2))
        #if detector output is not normalized
        #corners_values[:,0] = corners_values[:,0] * (W / PREDICTIONS_IMG_SHAPE[0])
        #corners_values[:,1] = corners_values[:,1] * (H / PREDICTIONS_IMG_SHAPE[1])
        #else
        corners_values[:,0] = corners_values[:,0] * W
        corners_values[:,1] = corners_values[:,1] * H
        
        if ADDGT:
            annotation = numpy.loadtxt(os.path.join(GT_CORNERS_FOLDER, '{}.txt'.format(idx))).flatten()
            corners_valuesgt = numpy.reshape(annotation, (9, 2))
            #if detector output is not normalized
            #corners_valuesgt[:,0] = corners_valuesgt[:,0] * (W / PREDICTIONS_IMG_SHAPE[0])
            #corners_valuesgt[:,1] = corners_valuesgt[:,1] * (H / PREDICTIONS_IMG_SHAPE[1])
            #else
            corners_valuesgt[:,0] = corners_valuesgt[:,0] * W
            corners_valuesgt[:,1] = corners_valuesgt[:,1] * H
            printed = draw_corners(image, corners_valuesgt, front=(0,255,0), back=(0,255,0), color=(0,255,0), size=6)    
            printed = draw_corners(printed, corners_values, front=(255,0,0), back=(255,0,0), color=(255,0,0), size=3)
        else:
            printed = draw_corners(image, corners_values, front=(255,0,0), back=(255,0,0), color=(255,0,0), size=3)
        
        video_writer.write(printed)
            
        cv2.imshow('Frame', printed)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'det__{}'.format(image_filename)), printed)
        
        key_ = cv2.waitKey(10)
        if key_ == 27:
            break
    
    video_writer.release()




if __name__ == '__main__':
    execute_()


