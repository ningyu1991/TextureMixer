import os
import os.path
from glob import glob
import numpy as np
import skimage
import skimage.io
import skimage.transform
import skimage.color
import math

import argparse

def get_image(path, path_ref, height, width, random_seed, transpose=True):
    #if 'earth' in path:
    #    image = histMatch(path, path_ref, random_seed)
    image = skimage.io.imread(path).astype(np.uint8)
    image = render(image, height, width, random_seed+1)
    image = transform(image, height, width, random_seed+2, transpose=transpose)
    return image

def transform(im0, height, width, random_seed, transpose):
    np.random.seed(random_seed)
    h = im0.shape[0]
    w = im0.shape[1]
    if h == height:
        i = 0
    else:
        i = np.random.randint(0, h-height)
    if w == width:
        j = 0
    else:
        j = np.random.randint(0, w-width)
    im1 = im0[i:i+height, j:j+width, :]
    if transpose:
        im1 = np.transpose(im1, [2, 0, 1])
    return im1

def histMatch(path, path_ref, random_seed):
    np.random.seed(random_seed)
    alpha = np.random.uniform()
    #if np.random.uniform(high=0.5) > alpha:
    #    alpha = 1.0 - alpha
    image = skimage.io.imread(path)
    idx1 = path.rfind('/')
    idx2 = path.rfind('.')
    path0 = path[:idx1]
    name1 = path[idx1+1:idx2]
    idx1 = path_ref.rfind('/')
    idx2 = path_ref.rfind('.')
    name2 = path_ref[idx1+1:idx2]
    if name1 == name2:
        return image
    path_histMatch = '%s_hist_match/%s_matchTo_%s.png' % (path0, name1, name2)
    image_histMatch = skimage.io.imread(path_histMatch)
    image = (1.0-alpha)*image + alpha*image_histMatch
    return image.astype(np.uint8)

def render(im0, height, width, random_seed):
    np.random.seed(random_seed)

    input_height = im0.shape[0]
    input_width = im0.shape[1]
    im0 = im0 / 255.0

    flag = True
    while flag:
        flag = False

        reflectence = np.random.randint(0, 4)
        if reflectence == 0:
            im1 = np.array(im0)
        elif reflectence == 1:
            im1 = np.fliplr(im0)
        elif reflectence == 2:
            im1 = np.flipud(im0)
        elif reflectence == 3:
            im1 = np.flipud(np.fliplr(im0))

        rotation = np.random.uniform(-90, 90)
        im2 = skimage.transform.rotate(im1, rotation, resize=True, mode='reflect')
        w_new, h_new = largest_rotated_rect(im1.shape[1], im1.shape[0], math.radians(rotation))
        im2 = crop_around_center(im2, w_new, h_new)

        log_scale = np.random.uniform(-np.log(4.0), 0.0)
        scale = np.exp(log_scale)
        #log_relscale = np.random.uniform(-np.log(1.3), np.log(1.3))
        #relscale = np.exp(log_relscale)
        relscale = 1.0
        im2 = skimage.transform.rescale(im2, (scale*relscale**0.5, scale/relscale**0.5), mode='reflect')
        if im2.shape[0] < height or im2.shape[1] < width:
            flag = True

    im2 = im2 * 255.0
    return im2.astype(np.uint8)

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.
  
    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a np / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--iPath', type=str, default=' ')
    parser.add_argument('--oPath', type=str, default=' ')
    parser.add_argument('--num_aug', type=int, default=1)
    args = parser.parse_args()

    if not os.path.isdir(args.oPath): os.mkdir(args.oPath)

    crop_height = 128
    crop_width = crop_height

    data = sorted(glob(os.path.join(args.iPath, '*.png')))
    num_images = len(data)
    num_repeats = args.num_aug // num_images + 1

    pivot = 0
    for (num, iFile) in enumerate(data[pivot:]):
        idx1 = iFile.rfind('/')
        idx2 = iFile.find('.png')
        name = iFile[idx1+1:idx2]
        for count in range(num_repeats):
            oFile = '%s/%s_aug%08d.png' % (args.oPath, name, count)
            if not os.path.isfile(oFile):
                refFile = np.random.choice(data)
                im_crop = get_image(iFile, refFile, crop_height, crop_width, random_seed=count*100000+(num+pivot)*10, transpose=False)
                skimage.io.imsave(oFile, im_crop)
            print('%d:%d, %d:%d' % (num_images, num+pivot, num_repeats, count), end='\r')

if __name__ == '__main__':
    main()