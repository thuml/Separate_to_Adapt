import numpy as np
import tensorflow as tf
import tensorlayer as tl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
import os
from collections import Counter
import matplotlib.pyplot as plt


class Accumulator(dict):
    def __init__(self, name_or_names, accumulate_fn=np.concatenate):
        super(Accumulator, self).__init__()
        self.names = [name_or_names] if isinstance(name_or_names, str) else name_or_names
        self.accumulate_fn = accumulate_fn
        for name in self.names:
            self.__setitem__(name, [])

    def updateData(self, scope):
        for name in self.names:
            self.__getitem__(name).append(scope[name])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb:
            print(exc_tb)
            return False

        for name in self.names:
            self.__setitem__(name, self.accumulate_fn(self.__getitem__(name)))

        return True
		
class TrainingModeManager:
    def __init__(self, nets, train=False):
        self.nets = nets 
        self.modes = [net.training for net in nets]
        self.train = train
    def __enter__(self):
        for net in self.nets:
            net.train(self.train)
    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for (mode, net) in zip(self.modes, self.nets):
            net.train(mode)
        self.nets = None # release reference, to avoid imexplicit reference
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True
		
def clear_output():
    def clear():
        return
    try:
        from IPython.display import clear_output as clear
    except ImportError as e:
        pass
    import os
    def cls():
        os.system('cls' if os.name == 'nt' else 'clear')

    clear()
    cls()

def addkey(diction, key, global_vars):
    diction[key] = global_vars[key]

def track_scalars(logger, names, global_vars):
    values = {}
    for name in names:
        addkey(values, name, global_vars)
    for k in values:
        values[k] = variable_to_numpy(values[k])
    for k, v in values.items():
        logger.log_scalar(k, v)
    print(values)

def variable_to_numpy(x):
    ans = x.cpu().data.numpy()
    if torch.numel(x) == 1:
        return float(np.sum(ans))
    return ans

def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))

def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
    ans = A + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    return float(ans)

def one_hot(n_class, index):
    tmp = np.zeros((n_class,), dtype=np.float32)
    tmp[index] = 1.0
    return tmp

class OptimWithSheduler:
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']
    def zero_grad(self):
        self.optimizer.zero_grad()
    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr = g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1
		
class OptimizerManager:
    def __init__(self, optims):
        self.optims = optims #if isinstance(optims, Iterable) else [optims]
    def __enter__(self):
        for op in self.optims:
            op.zero_grad()
    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None 
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True

def setGPU(i):
    global os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(i)
    gpus = [x.strip() for x in (str(i)).split(',')]
    NGPU = len(gpus)
    print('gpu(s) to be used: %s'%str(gpus))
    return NGPU

class Logger(object):
    def __init__(self, log_dir, clear=False):
        if clear:
            os.system('rm %s -r'%log_dir)
        tl.files.exists_or_mkdir(log_dir)
        self.writer = tf.summary.FileWriter(log_dir)
        self.step = 0
        self.log_dir = log_dir

    def log_scalar(self, tag, value, step = None):
        if not step:
            step = self.step
        summary = tf.Summary(value = [tf.Summary.Value(tag = tag,
                                                     simple_value = value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_images(self, tag, images, step = None):
        if not step:
            step = self.step
        
        im_summaries = []
        for nr, img in enumerate(images):
            s = StringIO()
            
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            
            if img.shape[-1] == 1:
                img = np.tile(img, [1, 1, 3])
            img = to_rgb_np(img)
            plt.imsave(s, img, format = 'png')

            img_sum = tf.Summary.Image(encoded_image_string = s.getvalue(),
                                       height = img.shape[0],
                                       width = img.shape[1])
            im_summaries.append(tf.Summary.Value(tag = '%s/%d' % (tag, nr),
                                                 image = img_sum))
        summary = tf.Summary(value = im_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_histogram(self, tag, values, step = None, bins = 1000):
        if not step:
            step = self.step
        values = np.array(values)
        counts, bin_edges = np.histogram(values, bins=bins)
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)
            
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_bar(self, tag, values, xs = None, step = None):
        if not step:
            step = self.step

        values = np.asarray(values).flatten()
        if not xs:
            axises = list(range(len(values)))
        else:
            axises = xs
        hist = tf.HistogramProto()
        hist.min = float(min(axises))
        hist.max = float(max(axises))
        hist.num = sum(values)
        hist.sum = sum([y * x for (x, y) in zip(axises, values)])
        hist.sum_squares = sum([y * (x ** 2) for (x, y) in zip(axises, values)])

        for edge in axises:
            hist.bucket_limit.append(edge - 1e-10)
            hist.bucket_limit.append(edge + 1e-10)
        for c in values:
            hist.bucket.append(0)
            hist.bucket.append(c)

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, self.step)
        self.writer.flush()

class AccuracyCounter:
    def __init__(self):
        self.Ncorrect = 0.0
        self.Ntotal = 0.0
        
    def addOntBatch(self, predict, label):
        assert predict.shape == label.shape
        correct_prediction = np.equal(np.argmax(predict, 1), np.argmax(label, 1))
        Ncorrect = np.sum(correct_prediction.astype(np.float32))
        Ntotal = len(label)
        self.Ncorrect += Ncorrect
        self.Ntotal += Ntotal
        return Ncorrect / Ntotal
    
    def reportAccuracy(self):
        return np.asarray(self.Ncorrect, dtype=float) / np.asarray(self.Ntotal, dtype=float)

def CrossEntropyLoss(label, predict_prob, class_level_weight = None, instance_level_weight = None, epsilon = 1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()
    
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)

def BCELossForMultiClassification(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon = 1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()
    
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    bce = -label * torch.log(predict_prob + epsilon) - (1.0 - label) * torch.log(1.0 - predict_prob + epsilon)
    return torch.sum(instance_level_weight * bce * class_level_weight) / float(N)
	
def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon= 1e-20):

    N, C = predict_prob.size()
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob*torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)

def plot_confusion_matrix(cm, true_classes,pred_classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools
    pred_classes = pred_classes or true_classes
    if normalize:
        cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    true_tick_marks = np.arange(len(true_classes))
    plt.yticks(true_classes, true_classes)
    pred_tick_marks = np.arange(len(pred_classes))
    plt.xticks(pred_tick_marks, pred_classes, rotation=45)


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
 
    if not true_labels:
        true_labels = sorted(list(set(list(y_true))))
    true_label_to_id = {x : i for (i, x) in enumerate(true_labels)}
    if not pred_labels:
        pred_labels = true_labels
    pred_label_to_id = {x : i for (i, x) in enumerate(pred_labels)}
    confusion_matrix = np.zeros([len(true_labels), len(pred_labels)])
    for (true, pred) in zip(y_true, y_pred):
        confusion_matrix[true_label_to_id[true]][pred_label_to_id[pred]] += 1.0
    return confusion_matrix