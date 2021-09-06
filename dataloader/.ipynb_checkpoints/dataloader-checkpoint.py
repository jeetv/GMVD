import os
import json
import numpy as np
from scipy.stats import multivariate_normal
from PIL import Image
from scipy.sparse import coo_matrix
from torchvision.datasets import VisionDataset
import torch
from torchvision.transforms import ToTensor

class GetDataset(VisionDataset):
    def __init__(self, base, reID=False, train=True, transform=ToTensor(), target_transform=ToTensor(),grid_reduce=4, img_reduce=4, train_ratio=0.9):
        # parameters in super can be accessed as (self.param) eg:- self.transform, self.target_transform
        super().__init__(base.root, transform=transform, target_transform=target_transform)
        self.base = base
        self.reID =  reID
        self.root, self.num_cam, self.num_frames = base.root, base.num_cam, base.num_frames
        self.img_shape, self.world_grid_shape = base.img_shape, base.world_grid_shape  # H,W; N_row,N_col
        
        # Reduce grid [480,1440] by factor of 4.
        self.grid_reduce = grid_reduce
        #[480,1440]/4 --> [120,360] i.e [1200cm,3600cm] and 10cm is pixel size, therefore [1200/10,3600/10]
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.world_grid_shape)) 

        self.img_reduce = img_reduce
        
        # Map kernel size = 41*41
        map_sigma, map_kernel_size = 20 / grid_reduce, 20
        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        # normalizing the kernel with max value.
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)
        
        # Split train/test data
        
        if train:
            frame_range = range(0,int(train_ratio*self.num_frames))
        else:
            frame_range = range(int(train_ratio*self.num_frames), self.num_frames)
        
        ############ Cam set Selection ############
        #frame_range = range(0,self.num_frames) 
        ###########################################
        self.img_fpath = self.base.get_image_paths(frame_range)
  
        # gt_map initialization 
        self.gt_map = {}
        self.download(frame_range)


    def download(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                i_s, j_s, v_s = [], [], []
                for single_pedestrian in all_pedestrians:
                    x, y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                    if self.base.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    v_s.append(single_pedestrian['personID'] + 1 if self.reID else 1)
                occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape)
                self.gt_map[frame] = occupancy_map

    def __getitem__(self, index):
        frame = list(self.gt_map.keys())[index]
        imgs = []
        for cam in range(self.num_cam):
            fpath = self.img_fpath[cam][frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        map_gt = self.gt_map[frame].toarray()
        if self.reID:
            map_gt = (map_gt > 0).int()
        if self.target_transform is not None:
            map_gt = self.target_transform(map_gt)
        return imgs, map_gt.float(), frame
    
    def __len__(self):
        # length of dataset
        return len(self.gt_map.keys())
