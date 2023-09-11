import cv2
from osgeo import gdal
import random
import numpy as np
import scipy.signal as sig


kx = np.array([[-1, 0, 1]])
ky = np.array([[-1], [0], [1]])


def readLeftImage(filename):
    dataset = gdal.Open(filename)
    if dataset is None:
        raise FileNotFoundError('Image file not found! Please check the path.')

    if dataset.RasterCount == 1:
        image = dataset.ReadAsArray()
        image = (image - np.mean(image)) / np.std(image)
        dx = sig.convolve2d(image, kx, 'same')
        dy = sig.convolve2d(image, ky, 'same')
        image = np.expand_dims(image.astype('float32'), -1)
        dx = np.expand_dims(dx.astype('float32'), -1)
        dy = np.expand_dims(dy.astype('float32'), -1)
        return image, dx, dy
    else:
        image, dx, dy = [], [], []
        for i in range(1, dataset.RasterCount + 1):
            band = dataset.GetRasterBand(i).ReadAsArray()
            band = band / 127.5 - 1.0
            bdx = sig.convolve2d(band, kx, 'same')
            bdy = sig.convolve2d(band, ky, 'same')
            image.append(band)
            dx.append(bdx)
            dy.append(bdy)
        image = np.transpose(np.array(image), [1, 2, 0]).astype('float32')
        dx = np.transpose(np.array(dx), [1, 2, 0]).astype('float32')
        dy = np.transpose(np.array(dy), [1, 2, 0]).astype('float32')
        return image, dx, dy


def readRightImage(filename):
    dataset = gdal.Open(filename)
    if dataset is None:
        raise FileNotFoundError('Image file not found! Please check the path.')

    if dataset.RasterCount == 1:
        image = dataset.ReadAsArray()
        image = (image - np.mean(image)) / np.std(image)
        return np.expand_dims(image.astype('float32'), -1)
    else:
        image = dataset.ReadAsArray()
        image = image / 127.5 - 1.0
        return np.transpose(image, [1, 2, 0]).astype('float32')


def readDisparity(filename):
    dataset = gdal.Open(filename)
    if dataset is None:
        raise FileNotFoundError('Image file not found! Please check the path.')
    if dataset.RasterCount != 1:
        raise ValueError('Disparity is error!')
    disparity = dataset.ReadAsArray()
    d16 = cv2.resize(disparity, (64, 64)) / 16.0
    d8 = cv2.resize(disparity, (128, 128)) / 8.0
    d4 = cv2.resize(disparity, (256, 256)) / 4.0
    disparity = np.expand_dims(disparity, -1)
    d16 = np.expand_dims(d16, -1)
    d8 = np.expand_dims(d8, -1)
    d4 = np.expand_dims(d4, -1)
    return d16, d8, d4, disparity


def readBatch(left_paths, right_paths, disp_paths):
    lefts, dxs, dys, rights, d16s, d8s, d4s, ds = [], [], [], [], [], [], [], []
    for left_path, right_path, disp_path in zip(left_paths, right_paths, disp_paths):
        left, dx, dy = readLeftImage(left_path)
        right = readRightImage(right_path)
        d16, d8, d4, d = readDisparity(disp_path)
        lefts.append(left)
        dxs.append(dx)
        dys.append(dy)
        rights.append(right)
        d16s.append(d16)
        d8s.append(d8)
        d4s.append(d4)
        ds.append(d)
    return np.array(lefts), np.array(rights), np.array(dxs), np.array(dys),\
           np.array(d16s), np.array(d8s), np.array(d4s), np.array(ds)


def loadBatch(all_left_paths, all_right_paths, all_disp_paths, batch_size=4, reshuffle=False):
    assert len(all_left_paths) == len(all_disp_paths)
    assert len(all_right_paths) == len(all_disp_paths)

    i = 0
    while True:
        lefts, rights, dxs, dys, d16s, d8s, d4s, ds = readBatch(
            left_paths=all_left_paths[i * batch_size:(i + 1) * batch_size],
            right_paths=all_right_paths[i * batch_size:(i + 1) * batch_size],
            disp_paths=all_disp_paths[i * batch_size:(i + 1) * batch_size])
        yield [lefts, rights, dxs, dys], [d16s, d8s, d4s, ds]
        i = (i + 1) % (len(all_left_paths) // batch_size)
        if reshuffle:
            if i == 0:
                paths = list(zip(all_left_paths, all_right_paths, all_disp_paths))
                random.shuffle(paths)
                all_left_paths, all_right_paths, all_disp_paths = zip(*paths)
