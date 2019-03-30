import argparse
import os
import shutil
import time
import sys
sys.path.insert(0, 'faster_rcnn')
import sklearn
import sklearn.metrics
import pdb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils

from tensorboardX import SummaryWriter

from datasets.factory import get_imdb
from custom import *

import matplotlib.pyplot as plt

from fast_rcnn.nms_wrapper import nms
import random

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=2,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=32,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.01,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--epoch-plot-freq',
    default=2,
    type=int,
    metavar='N',
    help='epoch plot frequency (default: 2)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', default=1,action='store_true')

best_prec1 = 0

class_names = ('aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor')

def convert_0_1(image):
    image_range = image.max() - image.min()
    if(image_range>0):
        image = (image - image.min())/image_range
    return image

def display_heatmap(image, s):
    heatmap = F.upsample(image.unsqueeze(0).unsqueeze(0), s)
    heatmap = convert_0_1(heatmap)
    return heatmap.squeeze()

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer
    criterion = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # TODO: Write code for IMDBDataset in custom.py
    trainval_imdb = get_imdb('voc_2007_trainval')
    test_imdb = get_imdb('voc_2007_test')

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    # Adding manual seed to force repeatability
    torch.manual_seed(5)

    normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    unnormalize = transforms.Normalize(mean = (-mean / std).tolist(), std=(1.0 / std).tolist())
    train_dataset = IMDBDataset(
        trainval_imdb,
        transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        IMDBDataset(
            test_imdb,
            transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    if args.evaluate:
        # validate(val_loader, model, criterion)

        for i, (input, target) in enumerate(val_loader):
        # pdb.set_trace()
        # Select only 20
            input = input[0:20]
            target = target[0:20]
            # Rest is similar
            for j in range(20):
                print("Evaluating Image "+str(j))
                t = target[j].type(torch.FloatTensor).cuda(async=True)
                output = model(input[j].unsqueeze(0))
                # vis.image(convert_0_1(input[j]) ,opts=dict(title='random_valid_'+str(j)))
                for index in t.nonzero():
                    ind = index.cpu().numpy()[0]
                    heatmapimage_ = output[0,ind]
                    heatmapimage_ = display_heatmap(heatmapimage_, input.size()[2:])
                    img = input[j].cpu().numpy()
                    img = np.transpose(img, (1,2,0))
                    heatmapimage_hots = (heatmapimage_.cpu().detach().numpy() > 0.95)*1
                    heatmapimage_hots_indices_r, heatmapimage_hots_indices_c = np.where(heatmapimage_hots==1)
                    dets = []
                    for samp in range(100):
                        r_1 = random.randint(0,heatmapimage_hots_indices_r.shape[0])
                        r_2 = random.randint(0,heatmapimage_hots_indices_r.shape[0])
                        c_1 = random.randint(0,heatmapimage_hots_indices_c.shape[0])
                        c_2 = random.randint(0,heatmapimage_hots_indices_c.shape[0])
                        score = len(np.where(heatmapimage_hots[r_1:r_2,c_1:c_2]==1)[0])
                        dets.append(np.array([c_1,c_2,r_1,r_2,score]))
                    dets = np.asarray(dets)
                    keep = nms(dets,0.5)
                    img_rect = nms_dets[keep]
                    cv2.rectangle(img,(img_rect[0],img_rect[1]),(img_rect[2],img_rect[3]),(0,255,0))
                    cv2.imwrite('results/valid_j_ind.png', img)
                # vis.heatmap( heatmapimage_.flip(0),opts=dict(title='random_valid_'+str(j)+'_heatmap_'+str(class_names[ind])))
        return

    # TODO: Create loggers for visdom and tboard
    # TODO: You can pass the logger objects to train(), make appropriate
    # modifications to train()
    if args.vis:
        import visdom
        vis = visdom.Visdom()
    writer = SummaryWriter()


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer, vis, unnormalize)

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, epoch, writer, vis, unnormalize)
            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
    
    # Display 20 randomly chosen images.
    for i, (input, target) in enumerate(val_loader):
        # pdb.set_trace()
        # Select only 20
        input = input[0:20]
        target = target[0:20]
        # Rest is similar
        for j in range(20):
            t = target[j].type(torch.FloatTensor).cuda(async=True)
            output = model(input[j].unsqueeze(0))
            vis.image(convert_0_1(input[j]) ,opts=dict(title='random_valid_'+str(j)))
            for index in t.nonzero():
                ind = index.cpu().numpy()[0]
                heatmapimage_ = output[0,ind]
                heatmapimage_ = display_heatmap(heatmapimage_, input.size()[2:])
                vis.heatmap( heatmapimage_.flip(0),opts=dict(title='random_valid_'+str(j)+'_heatmap_'+str(class_names[ind])))
        break



#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, args, writer, vis, unnormalize):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = input
        target_var = target

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        output = model(input)
        if(args.arch == 'localizer_alexnet_robust'):
            imoutput = F.avg_pool2d(output, kernel_size=output.size()[2:])
        else:
            imoutput = F.max_pool2d(output, kernel_size=output.size()[2:])
        imoutput = imoutput.squeeze()
        imoutput = torch.sigmoid(imoutput)
        loss = criterion(imoutput, target)

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.mean().item(), input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))

        # TODO:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print(losses)
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'
                .format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time
                      ,loss=losses
                      ,avg_m1=avg_m1
                      ,avg_m2=avg_m2
                      ))
        
        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
        n_iter = epoch*len(train_loader) + i
        # Plot the Training Loss
        writer.add_scalar('train/loss', loss.mean().item(), n_iter)
        writer.add_scalar('train/metric1', m1[0], n_iter)
        writer.add_scalar('train/metric2', m2[0], n_iter)
        writer.add_scalar('train/meanmetric1', avg_m1.avg, n_iter)
        writer.add_scalar('train/meanmetric2', avg_m2.avg, n_iter)

        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(tag, value.data.cpu().numpy(), n_iter)
            writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), n_iter)

        # To convert to a coloured CMAP
        cmap_converter = plt.get_cmap('inferno')
        # totensor = transforms.Compose([transforms.ToTensor]])

        # Plot images and heat maps of GT classes for 4 batches (2 images in each batch)
        if( i % 4 == 0 and i>0 and i<20):
            unnormalized_input = input[0:2]
            for i in range(unnormalized_input.shape[0]):
                unnormalized_input[i] = unnormalize(input[i])
            grid_images = torchvision.utils.make_grid(unnormalized_input)
            writer.add_image('Images', grid_images, n_iter)

            # HeatMap for second one
            for image_index in range(2):
                # pdb.set_trace()
                heatmapimages_color = []
                for index in target[image_index].nonzero():
                    ind = index.cpu().numpy()[0]
                    heatmapimage_ = output[image_index,ind]
                    heatmapimage_ = display_heatmap(heatmapimage_, input.size()[2:])
                    heatmapimage_color = transforms.ToTensor()(cmap_converter(heatmapimage_.cpu().detach().numpy())[:,:,:3]).float()
                    heatmapimages_color.append(heatmapimage_color)
                    if(epoch == 0 or epoch == (args.epochs-1)):
                        vis.heatmap( heatmapimage_.flip(0),opts=dict(title=str(epoch)+'_'+str(n_iter)+'_'+str(i)+'_image_'+str(image_index)+'_heatmap_'+str(class_names[ind])))
                heatmapimages_color = torch.stack(heatmapimages_color)
                grid_images = torchvision.utils.make_grid(heatmapimages_color)
                writer.add_image('HeatMap-Image'+str(image_index), grid_images, n_iter)

            # Same in Visdom with Title: <epoch>_<iteration>_<batch_index>_image, <epoch>_<iteration>_<batch_index>_heatmap_<class_name>
            if(epoch ==0 or epoch==(args.epochs-1)):
                vis.image( convert_0_1(input[0]),opts=dict(title=str(epoch)+'_'+str(n_iter)+'_'+str(i)+'_image_'+str(0)))
                vis.image( convert_0_1(input[1]),opts=dict(title=str(epoch)+'_'+str(n_iter)+'_'+str(i)+'_image_'+str(1)))

            # End of train()


def validate(val_loader, model, criterion, epoch, writer, vis, unnormalize):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = input
        target_var = target

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        imoutput = model(input)
        if(args.arch == 'localizer_alexnet_robust'):
            imoutput = F.avg_pool2d(imoutput, kernel_size=imoutput.size()[2:])
        else:
            imoutput = F.max_pool2d(imoutput, kernel_size=imoutput.size()[2:])
        imoutput = imoutput.squeeze()
        imoutput = torch.sigmoid(imoutput)
        loss = criterion(imoutput, target)


        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.mean().item(), input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'
                  .format(
                      i,
                      len(val_loader),
                      batch_time=batch_time
                      ,loss=losses
                      ,avg_m1=avg_m1
                      ,avg_m2=avg_m2
                      ))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
        n_iter = epoch*len(val_loader) + i
        if(epoch % args.epoch_plot_freq == 0):
            writer.add_scalar('test/loss', loss.mean().item(), n_iter)
            writer.add_scalar('test/meanmetric1', avg_m1.avg, n_iter)
            writer.add_scalar('test/meanmetric2', avg_m2.avg, n_iter)

        # if( i % 4 == 0 and i>0 and i<20):
        #     writer.add_image('Image', unnormalize(input[0]), n_iter)
        #     writer.add_image('Image', unnormalize(input[2]), n_iter)

        #     # HeatMap
            
        # # Same in Visdom with Title: <epoch>_<iteration>_<batch_index>_image, <epoch>_<iteration>_<batch_index>_heatmap_<class_name>
        #     vis.image( input[0],opts=dict(title='Test Image 1', caption='First of the two'))
        #     vis.image( input[2],opts=dict(title='Test Image 2', caption='Second'))





    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def metric1(output, target):
    # TODO: Ignore for now - proceed till instructed
    AP = compute_ap(target, output, np.ones(target.shape))
    # pdb.set_trace()
    mAP = np.mean(AP)
    return [mAP]


def metric2(output, target):
    #TODO: Ignore for now - proceed till instructed
    # output = F.sigmoid(output)
    output = output>0.5
    F_score = compute_f1(target, output)
    mF_score = np.mean(F_score)
    return [mF_score]


if __name__ == '__main__':
    main()
