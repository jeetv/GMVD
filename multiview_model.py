import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
#from MCMT.resnet import resnet18
import matplotlib.pyplot as plt
import random as rnd

class MultiView_Detection(nn.Module):
    def __init__(self, backbone_model, dataset, logdir, loss, avgpool, cam_set, len_cam_set):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.img_shape = dataset.img_shape
        self.reducedgrid_shape = dataset.reducedgrid_shape
        self.logdir = logdir
        self.avgpool = avgpool
        self.cam_set = cam_set
        
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                           dataset.base.extrinsic_matrices,
                                                                           dataset.base.worldgrid2worldcoord_mat)
        # Projection matrix
        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam)]
        
        # Coordinate Map
        self.coord_map = self.get_coord_map(self.reducedgrid_shape + [1])
        
        # Resnet18
        self.base_arch = nn.Sequential(*list(backbone_model.children())[:-2])
        '''
        for param in base_arch.parameters():
            param.requires_grad = False
        '''

        #split = 7
        #self.base_pt1 = base_arch[:split].to('cuda:1')
        #self.base_pt2 = base_arch[split:].to('cuda:0')
        if avgpool:
            self.out_channel = 512+2
        else:
            if self.cam_set:
                self.out_channel = 512 * len_cam_set + 2
            else:
                self.out_channel = 512 * self.num_cam + 2
        
        # Ground Plane Convolution
        if loss == 'klcc':
            #### for KLDiv+CC ####
            self.map_classifier = nn.Sequential(nn.Conv2d(self.out_channel, 512, 3, padding=1), nn.ReLU(),
                                                nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                                nn.Conv2d(512, 1, 3, padding=4, dilation=4), nn.Sigmoid())#.to('cuda:0')
            #############################################
        elif loss == 'mse':
            self.map_classifier = nn.Sequential(nn.Conv2d(self.out_channel, 512, 3, padding=1), nn.ReLU(),
                                                nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                                nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False))#.to('cuda:0')

        
    def forward(self, imgs, ignore_cam, duplicate_cam, random, cam_selected):
        B, N, C, H, W = imgs.shape
        if random: 
            if len(cam_selected)==0: 
                cam_selected = [i for i in range(N)]
                
            #Pick a new ignore camera for each sample print("Camera Ignored:",ignore_cam + 1)
            ignore_cam = rnd.choice(cam_selected)
            #Pick duplicate cam for concat operation
            duplicate_cam = random.choice([i for i in cam_selected if ignore_cam!=i])
           
        assert N == self.num_cam
        device = imgs.device
        world_features = []
        #world_features = torch.zeros(B,self.out_channel,self.reducedgrid_shape[0], self.reducedgrid_shape[1]).to(device)
        if self.cam_set:
            n_cam = len(cam_selected)
        else:
            n_cam = self.num_cam
            
        imgs_result = []
        for cam in range(n_cam):
            if self.cam_set:
                cam = cam_selected[cam]
            
            if random:
                #print('ignore : ',ignore_cam)
                if ignore_cam == cam:
                    if self.avgpool:
                        continue
                    else:
                        #print('duplicate : ',duplicate_cam)
                        cam = duplicate_cam
                
            # imgs[batch, cam, channel, h, w]
            #print(cam)
            img_feature = self.base_arch(imgs[:, cam])
            #img_feature = self.base_pt2(img_feature.to('cuda:0'))
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
            
            '''
            fig = plt.figure()
            subplt0 = fig.add_subplot(211, title="output")
            subplt0.imshow(torch.norm(img_feature[0].detach(), dim=0).cpu().numpy())
            plt.savefig('feature_map'+str(cam+1)+'.jpg')
            plt.close(fig)
            '''

            # 3x3 proj mat of cam --> repeat will make it [B,3,3]
            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to(device)
            if  self.avgpool:
                world_feature = kornia.warp_perspective(img_feature, proj_mat, self.reducedgrid_shape).unsqueeze(0)
            else:
                world_feature = kornia.warp_perspective(img_feature, proj_mat, self.reducedgrid_shape)
            ######### Concetenate features ###########
            world_features.append(world_feature)
            ##########################################
            #world_features += world_feature
            
        # torch.cat(list_of_tensors + list_of_tensor, dim=1)
        # torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1], dim=1)
        # world_features[B, num_camera*channel+2, H, W]
        #world_features /= self.num_cam
        #world_features = torch.cat([world_features] + [self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)
        #world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)
        if self.avgpool:
            world_features = torch.cat(world_features, dim=1)
            world_features = torch.mean(world_features, dim=1)    
            world_features = torch.cat([world_features] + [self.coord_map.repeat([B, 1, 1, 1]).to(device)], dim=1)
            
        else:
            world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to(device)], dim=1)
        
        
        fig = plt.figure()
        subplt0 = fig.add_subplot(211, title="output")
        subplt0.imshow(torch.norm(world_features[0].detach(), dim=0).cpu().numpy())
        plt.savefig(os.path.join(self.logdir,'world_features.jpg'))
        plt.close(fig)
        
        
        map_result = self.map_classifier(world_features)
        map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')
        return map_result
            
    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            # removing third column(z=0) from extrinsic matrix of size 3x4 
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)
            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)

            # transforming img axis to grid map axis
            # x(col),y(row) img coord -->  y(col), x(row) grid map coord
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
        return projection_matrices
    
    def get_coord_map(self, grid_shape):
        H, W, C = grid_shape
        # img[y,x] = img[h,w]
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        # making x and y in range [-1.0 to 1.0]
        grid_x = torch.from_numpy((grid_x / (W - 1) * 2) - 1).float()
        grid_y = torch.from_numpy((grid_y / (H - 1) * 2) - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        return ret
        
