import tensorflow as tf
import tensorflow.keras as keras


class CostVolume(keras.Model):
    def __init__(self, min_disp, max_disp, method):
        super(CostVolume, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.method = method

    def call(self, inputs, training=None, mask=None):
        assert len(inputs) == 2

        cost_volume = []
        # difference
        if self.method == 'diff':
            for i in range(self.min_disp, self.max_disp):
                if i < 0:
                    cost_volume.append(tf.pad(
                        tensor=inputs[0][:, :, :i, :] - inputs[1][:, :, -i:, :],
                        paddings=[[0, 0], [0, 0], [0, -i], [0, 0]], mode='CONSTANT'))
                elif i > 0:
                    cost_volume.append(tf.pad(
                        tensor=inputs[0][:, :, i:, :] - inputs[1][:, :, :-i, :],
                        paddings=[[0, 0], [0, 0], [i, 0], [0, 0]], mode='CONSTANT'))
                else:
                    cost_volume.append(inputs[0] - inputs[1])
        # concatenation
        elif self.method == 'concat':
            for i in range(self.min_disp, self.max_disp):
                if i < 0:
                    cost_volume.append(tf.pad(
                        tensor=tf.concat([inputs[0][:, :, :i, :], inputs[1][:, :, -i:, :]], -1),
                        paddings=[[0, 0], [0, 0], [0, -i], [0, 0]], mode='CONSTANT'))
                elif i > 0:
                    cost_volume.append(tf.pad(
                        tensor=tf.concat([inputs[0][:, :, i:, :], inputs[1][:, :, :-i, :]], -1),
                        paddings=[[0, 0], [0, 0], [i, 0], [0, 0]], mode='CONSTANT'))
                else:
                    cost_volume.append(tf.concat([inputs[0], inputs[1]], -1))
        else:
            raise TypeError('Method must be diff or concat!')

        cost_volume = tf.stack(cost_volume, 1)
        return cost_volume
