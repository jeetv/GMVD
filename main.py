import os
import time
import cv2
import numpy as np
from PIL import Image
import random
import argparse
import sys
import datetime
import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from dataloader.wildtrack import Wildtrack
from dataloader.multiviewx import MultiviewX
from dataloader.dataloader import GetDataset
from utils.logger import Logger
from utils.utils import loss_curve, nms, AverageMeter
from evaluation.evaluate import evaluate
from loss import Loss
from multiview_model import MultiView_Detection
from resnet import resnet18
from sklearn.metrics import precision_score, recall_score, accuracy_score
    
###################################################################################################################################
# Functions Train / Test
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def init_fn(worker_id):
    np.random.seed(int(args.seed))
    
def train(model, epoch, data_loader, optimizer, log_interval, scheduler=None):
    model.train()
    tic = time.time()
    precision_s = AverageMeter()
    recall_s = AverageMeter()
    accuracy_s = AverageMeter()
    losses, ignore_cam, duplicate_cam = 0, 0, 0
    if args.dropview:
        if args.cam_set:
            ignore_cam = random.choice(args.train_cam)
            duplicate_cam = random.choice([i for i in args.train_cam if ignore_cam!=i])
        else:
            ignore_cam = random.randint(0, data_loader.dataset.num_cam-1)
            duplicate_cam = random.choice([i for i in range(data_loader.dataset.num_cam) if ignore_cam!=i])
        print('Ignore cam : ', ignore_cam)
        if not args.avgpool:
            print('Duplicate cam : ', duplicate_cam)
    for batch_idx, (data, map_gt, _) in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        map_res = model(data, ignore_cam, duplicate_cam,args.dropview, args.train_cam)
        loss = criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel)
        loss.backward()
        optimizer.step()
        losses += loss.item()
        pred = (map_res > args.cls_thres).int().to(map_gt.device)

        true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
        false_positive = pred.sum().item() - true_positive
        false_negative = map_gt.sum().item() - true_positive
        precision = true_positive / (true_positive + false_positive + 1e-4)
        recall = true_positive / (true_positive + false_negative + 1e-4)
        '''
        precision = precision_score(map_gt.view(-1), pred.view(-1), zero_division=0)
        recall = recall_score(map_gt.view(-1), pred.view(-1))
        '''
        accuracy = accuracy_score(map_gt.view(-1), pred.view(-1))
        precision_s.update(precision)
        recall_s.update(recall)
        accuracy_s.update(accuracy)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {}, Batch:{}/{},\tLoss: {:.9f}, prec: {:.1f}%, recall: {:.1f}%, accuracy: {:.1f}%, \tTime: {:.3f} (min), maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), len(data_loader), losses / (batch_idx + 1), precision_s.avg * 100, recall_s.avg * 100, accuracy_s.avg * 100,
                    (time.time()-tic)/60, map_res.max()))
            pass

    print('Train Epoch: {}, Batch:{}, \tLoss: {:.9f}, Precision: {:.1f}%, Recall: {:.1f}%, accuracy: {:.1f}%, \tTime: {:.3f}(min)'.format(
            epoch, len(data_loader), losses / len(data_loader), precision_s.avg * 100, recall_s.avg * 100, accuracy_s.avg * 100, (time.time()-tic)/60))

    return losses / len(data_loader), precision_s.avg * 100, recall_s.avg * 100, accuracy_s.avg * 100


def test(model, epoch, data_loader, res_fpath=None, visualize=False):
    model.eval()
    tic = time.time()
    losses = 0
    precision_s = AverageMeter()
    recall_s = AverageMeter()
    accuracy_s = AverageMeter()
    all_res_list = []
    for batch_idx, (data, map_gt, frame) in enumerate(data_loader):
        data = data.to(device)
        
        with torch.no_grad():
            map_res = model(data, 0, 0, False, args.test_cam)
            
        if res_fpath is not None:
            map_grid_res = map_res.detach().cpu().squeeze()
            v_s = map_grid_res[map_grid_res > args.cls_thres].unsqueeze(1)
            grid_ij = torch.nonzero(map_grid_res > args.cls_thres)
            if data_loader.dataset.base.indexing == 'xy':
                grid_xy = grid_ij[:, [1, 0]]
            else:
                grid_xy = grid_ij
            all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                           data_loader.dataset.grid_reduce, v_s], dim=1))
            
        loss = criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel)
        losses += loss.item()
        pred = (map_res > args.cls_thres).int().to(map_gt.device)

        true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
        false_positive = pred.sum().item() - true_positive
        false_negative = map_gt.sum().item() - true_positive
        precision = true_positive / (true_positive + false_positive + 1e-4)
        recall = true_positive / (true_positive + false_negative + 1e-4)
        '''
        precision = precision_score(map_gt.view(-1), pred.view(-1), zero_division=0)
        recall = recall_score(map_gt.view(-1), pred.view(-1))
        '''
        accuracy = accuracy_score(map_gt.view(-1), pred.view(-1))
        precision_s.update(precision)
        recall_s.update(recall)
        accuracy_s.update(accuracy)

    if visualize and epoch!=0:
        fig = plt.figure()
        subplt0 = fig.add_subplot(211, title="output")
        subplt1 = fig.add_subplot(212, title="target")
        subplt0.imshow(map_res.detach().cpu().numpy().squeeze())
        subplt1.imshow(criterion.target_transform_(map_res, map_gt, data_loader.dataset.map_kernel).detach().cpu().numpy().squeeze())
        plt.tight_layout()
        plt.savefig(os.path.join(logdir, 'map_'+str(epoch)+'.jpg'))
        plt.close(fig)
        
    if res_fpath is not None:
        all_res_list = torch.cat(all_res_list, dim=0)
        np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
        res_list = []
        for frame in np.unique(all_res_list[:, 0]):
            res = all_res_list[all_res_list[:, 0] == frame, :]
            positions, scores = res[:, 1:3], res[:, 3]
            ids, count = nms(positions, scores, 20, np.inf)
            res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
        res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
        np.savetxt(res_fpath, res_list, '%d')
        
        recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath),
                                                    os.path.abspath(data_loader.dataset.base.gt_fname),
                                                    data_loader.dataset.base.__name__)
        print(f'moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}%')
    else:
        moda = 0
        
        
    print('Test, Loss: {:.9f}, Precision: {:.1f}%, Recall: {:.1f}, accuracy: {:.1f}%, \tTime: {:.3f}(min)'.format(
            losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, accuracy_s.avg * 100, (time.time()-tic)/60))
    
    return losses / len(data_loader), precision_s.avg * 100, recall_s.avg * 100, accuracy_s.avg * 100


###################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'], help='Choose dataset wildtrack/multiviewx (default: wildtrack)')
    parser.add_argument('-l', '--loss', type=str, default='klcc', choices=['klcc', 'mse'], help='Choose loss function klcc/mse. (default:klcc)' )
    parser.add_argument('-pr', '--pretrained', default=True, action='store_true', help='Use pretrained weights (default: True)')
    parser.add_argument('-cd', '--cross_dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=2, metavar='N', help='input batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--max_lr', type=float, default=1e-2, metavar='max LR', help=' max learning rate (default: 1e-2)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--p', type=float, default=1.0, help='hyper parameter to control CC loss')
    parser.add_argument('--k', type=float, default=1.0, help='hyper parameter to control KLDiv loss')
    parser.add_argument('--earlystop',  type=int, default=0, help='Store chkpt for particular epoch number (default: 0)')
    parser.add_argument('--avgpool', default=False, action='store_true', help='Enable average pooling (default: False)')
    parser.add_argument('--optim', type=str, default='SGD', choices=['SGD', 'Adam'], help='Choose optimizer.(default : SGD)')
    parser.add_argument('--step_size',default=1, type=int)
    parser.add_argument('--gamma',default=0.1, type=float)
    parser.add_argument('--lr_sched',type=str, default='onecycle_lr', choices=['step_lr', 'onecycle_lr'], help='Choose lr scheduler. (default:onecycle_lr)')
    parser.add_argument('--cam_set', default=False, action='store_true', help='Enable different camera set training and testing (default: False)')
    parser.add_argument("--train_cam", nargs="+", default=[])
    parser.add_argument("--test_cam", nargs="+", default=[])
    parser.add_argument('--dropview', default=False, action='store_true', help='Enable drop view training(default: False)')
    args = parser.parse_args()
    
    torch.cuda.empty_cache()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
        
    if args.loss=='klcc':
        args.cls_thres = sigmoid(args.cls_thres)
        args.lr = 1e-3
        args.max_lr = 1e-2
        args.momentum = 0.9
        
    args.train_cam = list(map(int, args.train_cam))
    args.test_cam = list(map(int, args.test_cam))
    
    #Logging
    logdir = f'logs/{args.dataset}_'+datetime.datetime.today().strftime('%d_%m_%Y_%H_%M')
    os.makedirs(logdir, exist_ok=True)
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))
    
    if args.cam_set:
        #if not args.avgpool:
        #    args.avgpool = True
        #    print('\nSetting {avgpool = True}\n')
        if len(args.train_cam) == 0 or len(args.test_cam) == 0:
            print('\nTrain/Test camera set is empty... Setting {cam_set = False}.\n')
            args.cam_set = False
            exit()
            
    args.train_cam = [x-1 for x in args.train_cam]
    print(args.train_cam)
    args.test_cam = [x-1 for x in args.test_cam]
    print(args.test_cam)
    
     # Dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize])
    
    if 'wildtrack' in args.dataset:
        data_path = os.path.expanduser('./Wildtrack')
        base = Wildtrack(data_path, args.cam_set, args.train_cam, args.test_cam)
    elif 'multiviewx' in args.dataset:
        data_path = os.path.expanduser('./MultiviewX')
        base = MultiviewX(data_path, args.cam_set, args.train_cam, args.test_cam)
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
        
    
    # Train and Test set
    train_dataset = GetDataset(base, train=True, transform=train_transform, grid_reduce=4, img_reduce=4)
    test_dataset = GetDataset(base, train=False, transform=train_transform, grid_reduce=4, img_reduce=4)
    
    # Train and Test Data Loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                            num_workers=args.num_workers, pin_memory=True, worker_init_fn=init_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True, worker_init_fn=init_fn)
    print(f'Train Data : {len(train_loader)*args.batch_size}')
    print(f'Test Data : {len(test_loader)}')
    
    # Model
    resnet_model = resnet18(pretrained=args.pretrained, replace_stride_with_dilation=[False, True, True])
    model = MultiView_Detection(resnet_model, train_dataset, logdir, args.loss, args.avgpool, args.cam_set, len(args.train_cam))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        print(torch.cuda.get_device_properties(device))
        model = nn.DataParallel(model)

    print(model)
    
    # Optimizer
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
        
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # LR Scheduler
    if args.lr_sched == 'step_lr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_sched == 'onecycle_lr':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, steps_per_epoch=len(train_loader),
                                                        epochs=args.epochs)
    else:
        scheduler = None
    
    # Loss
    criterion = Loss(args.loss, args.k, args.p).to(device)
    
    x_epoch = []
    train_loss_s = []
    test_loss_s = []
    train_prec_s = []
    test_prec_s = []
    train_recall_s = []
    test_recall_s = []
    train_acc_s = []
    test_acc_s = []
    #test_save_file = 'test_'+str(args.dataset)+'.txt'
    if args.resume is None:
        print('Testing....')
        test(model, 0, test_loader, res_fpath= os.path.join(logdir, 'test_'+str(args.dataset)+'_'+str(0)+'.txt'), visualize=True)
        
        for epoch in tqdm.tqdm(range(1, args.epochs+1)):
            print('Training...')
            train_loss, train_prec, train_recall, train_acc = train(model, epoch, train_loader, optimizer, args.log_interval, scheduler)
            
            print('Testing...')
            test_loss, test_prec, test_recall, test_acc = test(model, epoch, test_loader, os.path.join(logdir, 'test_'+str(args.dataset)+'_'+str(epoch)+'.txt'), visualize=True)
            
            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            test_loss_s.append(test_loss)
            train_prec_s.append(train_prec)
            test_prec_s.append(test_prec)
            train_recall_s.append(train_recall)
            test_recall_s.append(test_recall)
            train_acc_s.append(train_acc)
            test_acc_s.append(test_acc)

            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler.step()
            
            loss_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, test_loss_s, train_prec_s, test_prec_s, train_recall_s, test_recall_s,
                train_acc_s, test_acc_s)
    
            # Save Dictionary 
            #torch.save(model.state_dict(), os.path.join(logdir, 'Multiview_Detection_'+str(args.dataset)+'_'+str(epoch)+'.pth'))
            
            if args.earlystop!=0 and args.earlystop==epoch:
                #torch.save(model.state_dict(), os.path.join(logdir, 'Multiview_Detection_'+str(args.dataset)+'_'+str(epoch)+'.pth'))
                break
            #else :
        torch.save(model.state_dict(), os.path.join(logdir, 'Multiview_Detection_'+str(args.dataset)+'_'+str(epoch)+'.pth'))
        print("Training Completed..")
            

    else :
        resume_dir = f'logs/' + args.resume
        resume_fname = resume_dir #+ '/Multiview_Detection_'+str(args.cross_dataset)+'.pth'
        model.load_state_dict({k.replace('world_classifier','map_classifier'):v for k,v in torch.load(resume_fname).items()})
        model.eval()
        
    print('Test loaded model...')
    test(model, 1, test_loader, os.path.join(logdir, 'test_'+str(args.dataset)+'.txt'), visualize=True)
