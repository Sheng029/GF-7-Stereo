import glob
import os
import time
import numpy as np
import tensorflow.keras as keras
from feature import FeatureExtraction
from cost import CostVolume
from aggregation import Hourglass, FeatureFusion
from computation import Computation
from refinement import Refinement
from scheduler import schedule
from loss_functions import SmoothL1Loss
from data_loader import loadBatch, readLeftImage, readRightImage
from PIL import Image


class HMSMNet:
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
        gx = keras.Input(shape=(self.height, self.width, self.channel))
        gy = keras.Input(shape=(self.height, self.width, self.channel))

        feature_extraction = FeatureExtraction(filters=16)
        [l0, l1, l2] = feature_extraction(left_image)
        [r0, r1, r2] = feature_extraction(right_image)

        cost0 = CostVolume(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4, method='concat')
        cost1 = CostVolume(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8, method='concat')
        cost2 = CostVolume(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16, method='concat')
        cost_volume0 = cost0([l0, r0])
        cost_volume1 = cost1([l1, r1])
        cost_volume2 = cost2([l2, r2])

        hourglass0 = Hourglass(filters=16)
        hourglass1 = Hourglass(filters=16)
        hourglass2 = Hourglass(filters=16)
        agg_cost0 = hourglass0(cost_volume0)
        agg_cost1 = hourglass1(cost_volume1)
        agg_cost2 = hourglass2(cost_volume2)

        estimator2 = Computation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        disparity2 = estimator2(agg_cost2)

        fusion1 = FeatureFusion(units=16)
        fusion_cost1 = fusion1([agg_cost2, agg_cost1])
        hourglass3 = Hourglass(filters=16)
        agg_fusion_cost1 = hourglass3(fusion_cost1)

        estimator1 = Computation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        disparity1 = estimator1(agg_fusion_cost1)

        fusion2 = FeatureFusion(units=16)
        fusion_cost2 = fusion2([agg_fusion_cost1, agg_cost0])
        hourglass4 = Hourglass(filters=16)
        agg_fusion_cost2 = hourglass4(fusion_cost2)

        estimator0 = Computation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        disparity0 = estimator0(agg_fusion_cost2)

        # refinement
        refiner = Refinement(filters=32)
        final_disp = refiner([disparity0, left_image, gx, gy])

        self.model = keras.Model(inputs=[left_image, right_image, gx, gy],
                                 outputs=[disparity2, disparity1, disparity0, final_disp])
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
        mc = keras.callbacks.ModelCheckpoint(weights, 'val_refinement_loss', 1, True, True, 'min', 'epoch')

        # training
        optimizer = keras.optimizers.Adam()
        loss = [SmoothL1Loss(self.min_disp/16.0, self.max_disp/16.0),
                SmoothL1Loss(self.min_disp/8.0, self.max_disp/8.0),
                SmoothL1Loss(self.min_disp/4.0, self.max_disp/4.0),
                SmoothL1Loss(self.min_disp/1.0, self.max_disp/1.0)]
        loss_weights = [0.5, 0.7, 1.0, 0.6]
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)
        self.model.fit_generator(
            generator=loadBatch(train_left_paths, train_right_paths, train_dsp_paths, batch_size, True),
            steps_per_epoch=len(train_dsp_paths)//batch_size, epochs=epochs, callbacks=[tb, lr, mc],
            validation_data=loadBatch(val_left_paths, val_right_paths, val_dsp_paths, 2, False),
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
            left_image, dx, dy = readLeftImage(os.path.join(left_dir, left))
            right_image = readRightImage(os.path.join(right_dir, right))
            left_image = np.expand_dims(left_image, 0)
            gx = np.expand_dims(dx, 0)
            gy = np.expand_dims(dy, 0)
            right_image = np.expand_dims(right_image, 0)
            disparity = self.model.predict([left_image, right_image, gx, gy])[-1]
            disparity = Image.fromarray(disparity[0, :, :, 0])
            disparity.save(os.path.join(output_dir, left.replace('left', 'disparity')))
        t2 = time.time()
        print('Number of pairs: %d, total time: %.6f, average time: %.6f' % (len(lefts), t2 - t1, (t2 - t1)/len(lefts)))
