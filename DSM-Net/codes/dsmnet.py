import glob
import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from feature import FeatureExtractor
from cost import CostVolume
from aggregation import FactorizedCostAggregation
from computation import Computation
from refinement import Refinement
from scheduler import schedule
from loss_functions import SmoothL1Loss
from data_loader import loadBatch, readImage
from PIL import Image


class DSMNet:
    def __init__(self, height, width, channel, min_disp, max_disp):
        self.height = height
        self.width = width
        self.channel = channel
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.model = None

    def buildModel(self):
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))

        feature_extractor = FeatureExtractor(filters=16)
        [left_high_feature, left_low_feature] = feature_extractor(left_image)
        [right_high_feature, right_low_feature] = feature_extractor(right_image)

        high_cost_difference = CostVolume(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4, method='diff')
        low_cost_difference = CostVolume(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8, method='diff')
        high_cost_volume = high_cost_difference([left_high_feature, right_high_feature])
        low_cost_volume = low_cost_difference([left_low_feature, right_low_feature])

        low_aggregation = FactorizedCostAggregation(filters=16)
        low_agg_cost_volume = low_aggregation(low_cost_volume)

        low_computation = Computation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        low_disparity = low_computation(low_agg_cost_volume)  # 1/8

        upsample = keras.layers.UpSampling3D(size=(2, 2, 2))
        low_to_high = upsample(low_agg_cost_volume)
        high_cost_volume += low_to_high

        high_aggregation = FactorizedCostAggregation(filters=16)
        high_agg_cost_volume = high_aggregation(high_cost_volume)

        high_computation = Computation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        high_disparity = high_computation(high_agg_cost_volume)  # 1/4

        refine = Refinement(filters=16)
        refined_disparity = refine([high_disparity, left_image])

        d0 = tf.image.resize(low_disparity, [self.height, self.width]) * 8
        d1 = tf.image.resize(high_disparity, [self.height, self.width]) * 4
        d2 = tf.image.resize(refined_disparity, [self.height, self.width]) * 2

        self.model = keras.Model(inputs=[left_image, right_image], outputs=[d0, d1, d2])
        self.model.summary()

    def train(self, train_dir, val_dir, log_dir, weights, epochs, batch_size):
        # paths of dataset
        train_left_paths = glob.glob(train_dir + '/left/*')
        train_right_paths = glob.glob(train_dir + '/right/*')
        train_dsp_paths = glob.glob(train_dir + '/disparity/*')
        val_left_paths = glob.glob(val_dir + '/left/*')
        val_right_paths = glob.glob(val_dir + '/right/*')
        val_dsp_paths = glob.glob(val_dir + '/disparity/*')

        # sort
        train_left_paths.sort()
        train_right_paths.sort()
        train_dsp_paths.sort()
        val_left_paths.sort()
        val_right_paths.sort()
        val_dsp_paths.sort()

        # callbacks
        tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        lr = keras.callbacks.LearningRateScheduler(schedule=schedule, verbose=1)
        mc = keras.callbacks.ModelCheckpoint(weights, 'val_tf.math.multiply_2_loss', 1, True, True, 'min', 'epoch')

        # training
        optimizer = keras.optimizers.Adam()
        loss = [SmoothL1Loss(1.0 * self.min_disp, 1.0 * self.max_disp),
                SmoothL1Loss(1.0 * self.min_disp, 1.0 * self.max_disp),
                SmoothL1Loss(1.0 * self.min_disp, 1.0 * self.max_disp)]
        loss_weights = [0.8, 1.0, 0.6]
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)
        self.model.fit_generator(
            generator=loadBatch(train_left_paths, train_right_paths, train_dsp_paths, batch_size, 3, True),
            steps_per_epoch=len(train_dsp_paths)//batch_size, epochs=epochs, callbacks=[tb, lr, mc],
            validation_data=loadBatch(val_left_paths, val_right_paths, val_dsp_paths, 2, 3, False),
            validation_steps=len(val_dsp_paths)//2, shuffle=False)

    def predict(self, left_dir, right_dir, output_dir, weights):
        self.model.load_weights(filepath=weights, by_name=True)
        lefts = os.listdir(left_dir)
        rights = os.listdir(right_dir)
        lefts.sort()
        rights.sort()
        assert len(lefts) == len(rights)
        t1 = time.time()
        for left, right in zip(lefts, rights):
            left_image = np.expand_dims(readImage(os.path.join(left_dir, left)), 0)
            right_image = np.expand_dims(readImage(os.path.join(right_dir, right)), 0)
            disparity = self.model.predict([left_image, right_image])[-1]
            disparity = Image.fromarray(disparity[0, :, :, 0])
            disparity.save(os.path.join(output_dir, left.replace('left', 'disparity')))
        t2 = time.time()
        print('Number of pairs: %d, total time: %.6f, average: %.6f' % (len(lefts), t2 - t1, (t2 - t1) / len(lefts)))
