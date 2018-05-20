#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import math
import os
from os.path import exists, join, split
import time

import numpy as np
import shutil

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import pydensecrf.densecrf as dcrf

import drn
import data_transforms as transforms


try:
    from modules import batchnormsync
except ImportError:
    pass

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


HOUSE_PALETTE = np.array([[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 128, 0], [255, 0, 128],
                          [128, 255, 0], [128, 0, 255], [128, 128, 0], [128, 0, 128], [0, 128, 128], [255, 100, 100],
                          [100, 255, 100], [100, 100, 255]], dtype=np.uint8)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return self.softmax(y), x

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


def validate(val_loader, model, criterion, eval_score=None, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()
        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)[0]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        if eval_score is not None:
            score.update(eval_score(output, target_var), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {score.val:.3f} ({score.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                score=score))

    logger.info(' * Score {top1.avg:.3f}'.format(top1=score))

    return losses.avg, score.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """ Computes the precision@k for the specified values of k """
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    score = correct.float().sum(0).mul(100.0 / correct.size(0))
    return score.data[0]


def train(train_loader, model, criterion, optimizer, epoch,
          eval_score=None, print_freq=10):
    """ Train a single epoch. """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()

        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)[0]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        if eval_score is not None:
            scores.update(eval_score(output, target_var), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, loss=losses, top1=scores))
    return losses.avg, scores.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        out_dirname = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(out_dirname, 'model_best.pth.tar'))


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind][:-4] + '_res.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def save_colored_output(predictions, image, filenames, output_dir, palettes, suffix=''):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    for ind in range(len(filenames)):
        if not exists(output_dir):
            os.makedirs(output_dir)
        input_image = Image.fromarray(image[ind])
        input_image.save(os.path.join(output_dir, filenames[ind][:-4] + '.jpg'))
        label_array = predictions[ind].astype(np.int32)
        # label_image = Image.fromarray(label_array)
        # label_name = os.path.join(output_dir, filenames[ind][:-4] + suffix + '.pgm')
        # label_image.save(label_name)
        vis_image = Image.fromarray(palettes[label_array])
        vis_name = os.path.join(output_dir, filenames[ind][:-4] + suffix + '.png')
        vis_image.save(vis_name)


def refine_by_crf(image_batch, network_out, iter=50):
    """
    Clean the segmentation boundary by CRF.
    :param image_batch: the current batch of the image.
    :param network_out: the corrent batch of the network output.
    :param iter: maximum iterations for CRF.
    :return: refined segmentation mask.
    """
    assert image_batch.shape[0] == network_out.shape[0]
    n_label = network_out.shape[1]
    pred_batch = np.zeros([network_out.shape[0], network_out.shape[-2], network_out.shape[-1]], dtype=np.int64)
    neg_p_batch = -1 * network_out.cpu().data.numpy()
    for ind in range(image_batch.shape[0]):
        crf = dcrf.DenseCRF2D(image_batch[ind].shape[1], image_batch[ind].shape[0], n_label)
        crf.setUnaryEnergy(neg_p_batch.reshape(n_label, -1))
        crf.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)
        crf.addPairwiseBilateral(sxy=(10, 10), srgb=(13, 13, 13), rgbim=image_batch[ind], compat=10,
                                 kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        crf_out = crf.inference(iter)
        pred_batch[ind] = np.argmax(crf_out, axis=0).reshape(image_batch[ind].shape[:2]).astype(np.int64)
    return pred_batch


def test(eval_data_loader, model, num_classes,
         output_dir='pred', has_gt=True, run_crf=False, save_vis=False):
    """
    Run testing with a trained model. The function optionally refine the segmentation mask by fully-connected CRF.
    """
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    t_normalize = eval_data_loader.dataset.transforms.transforms[-1]

    for iter, (image, label, name) in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        image_var = Variable(image, requires_grad=False, volatile=True)
        final = model(image_var)[0]

        image_cpu = t_normalize.scale_back(image).transpose(1, 2).transpose(2, 3).contiguous().numpy()
        image_cpu = (image_cpu * 255).astype(np.uint8)

        suffix = ''
        if run_crf:
            # Run fully connected CRF to clear the boundary
            suffix = '_crf_50'
            pred_batch = refine_by_crf(image_cpu, final)
        else:
            _, pred_batch = torch.max(final, 1)
            pred_batch = pred_batch.cpu().data.numpy()

        batch_time.update(time.time() - end)
        if save_vis:
            save_colored_output(pred_batch, image_cpu, name, output_dir, HOUSE_PALETTE, suffix)
        if has_gt:
            label = label.numpy()
            print(pred_batch.dtype, label.dtype)
            hist += fast_hist(pred_batch.flatten(), label.flatten(), num_classes)
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)
