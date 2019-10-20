import os
import keras
from copy import deepcopy
from keras.callbacks import Callback
from keras import backend as K
import saliency


class EpochSaliency(keras.callbacks.Callback) :
    def __init__(self, conf, sample, label):
        self.conf = conf
        self.sample = sample.copy()
        print("init={}".format(self.sample.shape))
        self.label = label

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_batch_begin(self, batch, logs = None) :
        pass

    def on_batch_end(self, batch, logs = None) :
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs = None) :
        print("ini2t={}".format(self.sample.shape))
        sali = saliency.calculate_saliency_with_vis(self.conf, self.sample, self.label, self.model)
        epoch_dir = self.conf['paths']['saliency_dir_path'] + 'epoch/'
        if not os.path.exists(epoch_dir):
            os.mkdir(epoch_dir)
        path = epoch_dir + str(epoch) + '.png'
        saliency.generate_heatmap(self.conf, sali, self.sample, epoch, path=path)


class GifSaliency(keras.callbacks.Callback) :
    def __init__(self, conf, sample, label):
        self.conf = conf
        self.sample = sample.copy()
        print("init={}".format(self.sample.shape))
        self.label = label
        self.saliency_list = []

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        saliency.generate_animation_heatmap(self.conf, self.saliency_list, self.sample)

    def on_batch_begin(self, batch, logs = None) :
        pass

    def on_batch_end(self, batch, logs = None) :
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs = None) :
        sali = saliency.calculate_saliency_with_vis(self.conf, self.sample, self.label, self.model)
        self.saliency_list.append(sali)
