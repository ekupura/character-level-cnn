import os
import keras
from copy import deepcopy
from keras.callbacks import Callback
from keras import backend as K
from tqdm import tqdm
import saliency
import pickle


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
    def __init__(self, conf, samples, labels, gif=True):
        self.conf = conf
        self.samples = samples
        self.labels = labels
        self.gif = gif
        self.saliencies = [[] for i in samples]

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        print("generating...")
        if self.gif:
            for case, (sali, sample, label) in tqdm(enumerate(zip(self.saliencies, self.samples, self.labels))):
                saliency.generate_animation_heatmap(self.conf, case, sali, sample, label)
        else:
            with open(self.conf["paths"]["saliency_pkl_path"], "wb") as f:
                pickle.dump((list(range(len(self.samples))), self.saliencies, self.samples, self.labels), f, protocol=4)


    def on_batch_begin(self, batch, logs = None):
        pass

    def on_batch_end(self, batch, logs = None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        print("calculating...")
        saliencies = saliency.calculate_saliency_multi(self.conf, self.samples, self.labels, self.model)
        for i, s in enumerate(saliencies):
            self.saliencies[i].append(s)
