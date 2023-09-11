from osgeo import gdal
import random
import numpy as np


def readImage(filename):
    '''
    Read an image and standardize it.
    :param filename: path of image file.
    :return: a standardized image with shape [H, W, C].
    '''
    dataset = gdal.Open(filename)
    if dataset is None:
        raise FileNotFoundError('Image file not found! Please check the path.')

    raster_count = dataset.RasterCount
    if raster_count == 1:   # single-band image
        image = dataset.ReadAsArray()
        image = (image - np.mean(image)) / np.std(image)
        return np.expand_dims(image.astype('float32'), -1)   # [H, W, 1]
    else:
        image = dataset.ReadAsArray()
        image = image / 127.5 - 1.0
        return np.transpose(image, [1, 2, 0]).astype('float32')   # [H, W, C]


def readDisparity(filename):
    '''
    Read a disparity map.
    :param filename: path of disparity file.
    :return: a disparity map with shape [H, W, 1].
    '''
    dataset = gdal.Open(filename)
    if dataset is None:
        raise FileNotFoundError('Image file not found! Please check the path.')
    if dataset.RasterCount != 1:
        raise ValueError('Disparity is error!')
    disparity = dataset.ReadAsArray()
    return np.expand_dims(disparity, -1).astype('float32')   # [H, W, 1]


def readBatch(left_paths, right_paths, dsp_paths):
    '''
    Read a batch of left images, right images, and disparity maps.
    :param left_paths: paths of left image files.
    :param right_paths: paths of right image files.
    :param dsp_paths: paths of disparity files.
    :return: left images, right images, and disparity maps with shape [B, H, W, C], [B, H, W, C], and [B, H, W, 1]
    '''
    left_images, right_images, dsp_maps = [], [], []
    for left_path, right_path, dsp_path in zip(left_paths, right_paths, dsp_paths):
        left_images.append(readImage(left_path))
        right_images.append(readImage(right_path))
        dsp_maps.append(readDisparity(dsp_path))
    return np.array(left_images), np.array(right_images), np.array(dsp_maps)


def loadBatch(all_left_paths, all_right_paths, all_dsp_paths, batch_size, num_output, reshuffle=False):
    '''
    Generator for training pipeline.
    :param all_left_paths: paths of all left image files.
    :param all_right_paths: paths of all right image files.
    :param all_dsp_paths: paths of all left disparity files.
    :param reshuffle: whether to disrupt the order of files after an epoch.
    :return:
    '''
    assert len(all_left_paths) == len(all_dsp_paths)
    assert len(all_right_paths) == len(all_dsp_paths)

    i = 0
    while True:
        left_images, right_images, dsp_maps = readBatch(
            all_left_paths[i*batch_size:(i+1)*batch_size],
            all_right_paths[i*batch_size:(i+1)*batch_size],
            all_dsp_paths[i*batch_size:(i+1)*batch_size])
        if num_output == 1:
            yield [left_images, right_images], dsp_maps
        else:
            yield [left_images, right_images], [dsp_maps] * num_output
        i = (i + 1) % (len(all_dsp_paths) // batch_size)
        if reshuffle and i == 0:
            paths = list(zip(all_left_paths, all_right_paths, all_dsp_paths))
            random.shuffle(paths)
            all_left_paths, all_right_paths, all_dsp_paths = zip(*paths)
