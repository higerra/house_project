import os
import sys
sys.path.append('drn')

import shutil
import logging
import numpy as np
import json
from PIL import Image
import cv2
import torch

import segment
import data_transforms

# The class map for the current dataset. As the dataset growing, you might need to add additional class.
class_map = {'background': 0, 'Masonry': 6, 'Trim': 3, 'Corner Trim': 10, 'Door': 8, 'Window': 7,
             'Shutter': 9, 'Siding Main': 2, 'Siding Accent': 1, 'Roofing': 4, 'Foreground': 5, 'Siding': 11}

color_table = np.array([[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 128, 0], [255, 0, 128],
                        [128, 255, 0], [128, 0, 255], [128, 128, 0], [128, 0, 128], [0, 128, 128], [255, 100, 100],
                        [100, 255, 100], [100, 100, 255]])

_ignore_idx = -1


class HouseDataList(torch.utils.data.Dataset):
    """
    This class manages the file list of images and ground truth annotations. "phase" should be one of "train",
    "validation" and "test". The full path of a specific sample is the concatentation of "data_dir", "phase" and
    sample name.
    """
    def __init__(self, data_dir, phase, transforms, out_name=False):
        self.data_path = os.path.join(data_dir, phase)
        self.transforms = transforms
        self.image_list = []
        self.label_list = []
        self.out_name = out_name
        with open(os.path.join(self.data_path, 'dataset.json'), 'r') as f:
            self.dataset_info = json.load(f)
        for sample in self.dataset_info['samples']:
            self.image_list.append(sample['image'])
            if 'annotation' in sample:
                self.label_list.append(sample['annotation'])

    def has_label(self):
        """
        For inference, there is no ground truth label.
        """
        return len(self.label_list) == len(self.image_list)
    
    def __getitem__(self, index):
        """
        This function load the image and ground truth (for training and validation) for a specific sample. The
        function optionally return the sample name for debugging and visualization purpose.
        """
        data = [Image.open(os.path.join(self.data_path, self.image_list[index]))]
        if self.has_label():
            data.append(np.load(os.path.join(self.data_path, self.label_list[index])))
        data = list(self.transforms(*data))
        if self.out_name:
            if not self.has_label():
                data.append(np.zeros([data[0].height, data[0].width()]))
            data.append(self.image_list[index])
        return tuple(data)
        
    def __len__(self):
        return len(self.image_list)


""" The following functions defines some additional transformations. """

class RandomHorizontalFlip(object):
    def __init(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, label):
        if np.random.random() < self.prob:
            results = [image.transpose(Image.FLIP_LEFT_RIGHT),
                      np.fliplr(label)]
        else:
            results = [image, label]
        return results


class RescaleToFixedSize(object):
    """
    The semantic segmentation network does not require the image size to be fixed. However, during the training phase,
    samples inside the same batch need to have the same size to form a tensor. Therefore we rescale the images to the
    same size during training.
    """
    def __init__(self, target_size):
        # target_size: (width, height)
        self.target_size = target_size

    def __call__(self, image, label):
        w, h = image.size
        if w == self.target_size[0] and h == self.target_size[1]:
            return image, label
        if self.target_size[0] < w or self.target_size[1] < h:
            interpolation = Image.ANTIALIAS
        else:
            interpolation = Image.CUBIC
        resized_label = cv2.resize(label, dsize=self.target_size, interpolation=cv2.INTER_NEAREST)
        return image.resize(self.target_size, interpolation), resized_label


class IgnoreUnlabelledPixels(object):
    """
    The ground might contain a large number of unlabelled pixels. To reduce the confusion these pixels caused to the
    network, we mark all background pixels within the bounding box of non-background pixels as "ignored".
    """
    def __call__(self, image, label):
        c_hull = cv2.convexHull(cv2.findNonZero((label > 0).astype(np.uint8))).squeeze()
        new_label = np.copy(label)
        cv2.drawContours(new_label, [c_hull], 0, _ignore_idx, -1)
        new_label[label > 0] = label[label > 0]
        return image, new_label


FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TrainingOption:
    def __init__(self, data_dir, arch, num_classes, batch_size, learning_rate, lr_mode,
                 momentum, weight_decay, epochs, step, pretrained, num_workers):
        self.data_dir = data_dir
        self.input_size = (720, 640)
        self.arch = arch
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr_mode = lr_mode
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.step = step
        self.pretrained = pretrained
        self.num_workers = num_workers
        
        
def adjust_learning_rate(options, optimizer, current_epoch):
    if options.lr_mode == 'step':
        lr = options.learning_rate * (0.1 ** (current_epoch // options.step))
    elif options.lr_mode == 'poly':
        lr = options.learning_rate * (1 - current_epoch / options.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode:{}'.format(options.lr_mode))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    

def train_house(options, model_dir=None, resume_path=None):
    print('Building model...')
    single_model = segment.DRNSeg(options.arch, options.num_classes, None, pretrained=False)
    if options.pretrained:
        single_model.load_state_dict(torch.load(options.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()
    criterion = torch.nn.NLLLoss2d(ignore_index=_ignore_idx)
    criterion.cuda()
    
    # Data loading
    print('Loading data')
    with open(os.path.join(options.data_dir, 'info.json')) as f:
        info_json = json.load(f)
    t_normalize = data_transforms.Normalize(mean=info_json['mean'],
                                            std=info_json['std'])
    t_rescale = RescaleToFixedSize(options.input_size)
    transforms = [t_rescale, RandomHorizontalFlip(), IgnoreUnlabelledPixels(), data_transforms.ToTensor(), t_normalize]
    train_loader = torch.utils.data.DataLoader(
        HouseDataList(options.data_dir, 'train', data_transforms.Compose(transforms)),
        batch_size=options.batch_size, shuffle=True, num_workers=options.num_workers,
        pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        HouseDataList(options.data_dir, 'validation', data_transforms.Compose(transforms)),
        batch_size=options.batch_size, shuffle=True, num_workers=options.num_workers,
        pin_memory=True, drop_last=True
    )
    
    # Define loss function (critierion) and optimizer
    print('Setting up optimizer')
    optimizer = torch.optim.SGD(single_model.optim_parameters(), options.learning_rate,
                                options.momentum, options.weight_decay)
    torch.backends.cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    train_loss_score = []
    validation_loss_score = []

    if resume_path:
        if os.path.isfile(resume_path):
            print('Loading checkpoint {}'.format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print('Checkpoint {} loaded'.format(resume_path))
        else:
            print('Invalid checkpoint file path.')
    for epoch in range(start_epoch, options.epochs):
        lr = adjust_learning_rate(options, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        train_loss, train_score = segment.train(train_loader, model, criterion, optimizer, epoch,
                                                eval_score=segment.accuracy)
        val_loss, val_score = segment.validate(val_loader, model, criterion, eval_score=segment.accuracy)

        train_loss_score.append([train_loss, train_score])
        validation_loss_score.append([val_loss, val_score])

        if model_dir:
            is_best = val_score > best_prec1
            best_prec1 = max(val_score, best_prec1)
            checkpoint_path = os.path.join(model_dir, 'checkpoint_latest.pth.tar')
            segment.save_checkpoint({
                'epoch': epoch + 1,
                'arch': options.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1
            }, is_best, filename=checkpoint_path)
            print('Checkpoint for epoch {} saved.'.format(epoch + 1))
            if (epoch + 1) % 10 == 0:
                history_path = os.path.join(model_dir, 'checkpoint_{:03d}.pth.tar'.format(epoch + 1))
                print('Checkpoint file saved to ' + history_path)
                shutil.copyfile(checkpoint_path, history_path)
    if model_dir:
        # Save the training history
        train_loss_score = np.array(train_loss_score)
        validation_loss_score = np.array(validation_loss_score)
        np.savetxt(os.path.join(model_dir, 'validation_loss_score.txt'), validation_loss_score)
        np.savetxt(os.path.join(model_dir, 'train_loss_score.txt'), train_loss_score)


def test_house(options, model_dir, data_dir, phase, run_crf, out_dirname):
    single_model = segment.DRNSeg(options.arch, len(class_map), pretrained_model=None,
                                  pretrained=False)
    checkpoint = torch.load(os.path.join(model_dir, 'model_best.pth.tar'))
    model = torch.nn.DataParallel(single_model)
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded')
    model.cuda()

    with open(os.path.join(data_dir, 'info.json'), 'r') as f:
        info_json = json.load(f)
    t_normalize = data_transforms.Normalize(mean=info_json['mean'],
                                            std=info_json['std'])
    t_rescale = RescaleToFixedSize(options.input_size)
    transforms = [t_rescale, data_transforms.ToTensor(), t_normalize]
    # transforms = [data_transforms.ToTensor(), t_normalize]

    test_loader = torch.utils.data.DataLoader(
        HouseDataList(data_dir, phase, data_transforms.Compose(transforms), out_name=True),
        batch_size=1, shuffle=False, num_workers=options.num_workers,
        pin_memory=True, drop_last=True
    )
    torch.backends.cudnn.benchmark = True
    # Make the output directory
    out_dir = os.path.join(data_dir, out_dirname)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    mAP = segment.test(test_loader, model, options.num_classes, save_vis=True, has_gt=True, run_crf=run_crf,
                       output_dir=out_dir)
    logger.info('mAP: {}'.format(mAP))

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--data_dir', type=str,
                        default='../data/dataset1280')
    parser.add_argument('--model_dir', type=str, default='../model/test_model')
    parser.add_argument('--out_dirname', type=str, default='test_vis')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='The path to a saved model from which the training will be resumed.')
    parser.add_argument('--arch', type=str, default='drn_c_26')
    parser.add_argument('--batch_size', type=int, default=8,
                        help="The batch size. Larger batch size requires more GPU memory.")
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--run_crf', action='store_true')

    args = parser.parse_args()

    # Create the output folder if not exists
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)

    options = TrainingOption(args.data_dir, arch=args.arch, num_classes=len(class_map), batch_size=args.batch_size,
                             learning_rate=0.01, lr_mode='step', momentum=0.9, weight_decay=1e-4,
                             epochs=args.epochs, step=200, pretrained=False, num_workers=args.num_workers)

    if args.mode == 'train':
        train_house(options, args.model_dir, args.resume_path)
    elif args.mode == 'test':
        test_house(options, args.model_dir, args.data_dir, 'test', args.run_crf, args.out_dirname)
    else:
        raise ValueError('Unrecognized mode: ', args.mode)
