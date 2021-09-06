import numpy as np
import os
import sys
import cv2
import re
import json
from scipy.sparse import coo_matrix
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset

intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
                                     'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
                                     'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']

class Wildtrack(VisionDataset):
    def __init__(self, root, cam_set, train_cam, test_cam):
        super().__init__(root)
        self.root = root
        self.gt_fname = os.path.join(self.root,'gt.txt')
        #self.prepare_gt()
        self.dataset_name = 'Wildtrack'
        self.__name__ = 'Wildtrack'
        # shape[h,w]
        with open(os.path.join(self.root,'config.json'), 'r') as f:
            config = json.load(f)
        self.num_cam, self.num_frames = config['num_cam'], config['num_frames']
        self.img_shape, self.world_grid_shape = config['img_shape'], config['grid_shape']
        self.grid_cell, self.origin = config['grid_cell'], config['origin']
        # eg- region_size = 12m*36m (in meters) 
        self.region_size = config['region_size'] 
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])
        self.worldgrid2worldcoord_mat = np.array([[2.5, 0, self.origin[0]], [0, 2.5, self.origin[1]], [0, 0, 1]])
        self.indexing = 'ij'
        print(self.root)
        print(f'Dataset Name : {self.dataset_name}')
        print(f'Cameras : {self.num_cam}, Frames : {self.num_frames}')
        print(f'Image Shape(H,W) : {self.img_shape}')
        print(f'Grid Shape(rows,cols) : {self.world_grid_shape}')
        print(f'Grid Cell(in cm) : {self.grid_cell}cm i.e {self.grid_cell/100}m')
        print(f'Grid Origin(x,y) : {self.origin}')
        print(f'Area/Region size(in m) : {self.region_size[0]}m x {self.region_size[1]}m')
        self.prepare_gt()
        self.cam_set = cam_set
        if self.cam_set:
            self.train_cam = np.array(train_cam)
            self.test_cam = np.array(test_cam)
            self.bbox_by_pos_cam = self.read_POM2()
            self.overlapping_pos = self.final_overlap_pos()
        
    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero')
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             intrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()

        extrinsic_params_file_root = ET.parse(os.path.join(self.root, 'calibrations', 'extrinsic',
                                                           extrinsic_camera_matrix_filenames[camera_i])).getroot()

        rvec = extrinsic_params_file_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
        rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)

        tvec = extrinsic_params_file_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')
        tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix

    '''
    def read_POM(self):
        filename = 'rectangles.pom'
        bbox_by_pos_cam = {}
        cam_pos_pattern = re.compile(r'(\d+) (\d+)')
        cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
        with open(os.path.join(self.root,filename),'r') as f:
            for line in f:
                if 'RECTANGLE' in line:
                    cam, pos = map(int,cam_pos_pattern.search(line).groups())
                    if pos not in bbox_by_pos_cam:
                        bbox_by_pos_cam[pos] = {}
                    if 'notvisible' in line:
                        bbox_by_pos_cam[pos][cam] = None
                    else:
                        cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                        bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0), min(right, 1920 - 1), min(bottom, 1080 - 1)]
                        
        return bbox_by_pos_cam
    '''
    def get_worldgrid_from_pos(self, pos):
        R,C = self.world_grid_shape
        grid_x = pos % R
        grid_y = pos // R
        # [0,0]...[479,0],[0,1]..[479,1]...
        return np.array([grid_x, grid_y], dtype=int)
    
    def get_worldcoord_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        coord_x = self.origin[0] + self.grid_cell * grid_x  # -300 + 2.5 * x
        coord_y = self.origin[1] + self.grid_cell * grid_y  # -900 + 2.5 * x
        return np.array([coord_x, coord_y])
    
    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)
    
    def get_pos_from_worldgrid(self, worldgrid):
        R,C = self.world_grid_shape
        grid_x, grid_y = worldgrid
        pos = grid_x + grid_y * R
        return pos
    
    def get_worldgrid_from_worldcoord(self, worldcoord):
        coord_x, coord_y = worldcoord
        grid_x = (coord_x - self.origin[0]) / self.grid_cell  # (cx + 300) / 2.5 
        grid_y = (coord_y - self.origin[1]) / self.grid_cell  # (cy + 900) / 2.5 
        return np.array([grid_x, grid_y], dtype=int)
    
    def get_pos_from_worldcoord(self, worldcoord):
        grid = self.get_worldgrid_from_worldcoord(worldcoord)
        return self.get_pos_from_worldgrid(grid)
    
    def get_image_paths(self, frame_range):    
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)   
        return img_fpaths
    
    def prepare_gt(self):
        gt_dir = os.path.join(self.root, 'annotations_positions')
        gt_filenames = next(os.walk(gt_dir))[2]
        gt_out = []
        for i in range(len(gt_filenames)):
            frame_num = int(gt_filenames[i].split('.')[0])
            with open((os.path.join(gt_dir,gt_filenames[i]))) as json_file:
                gt_file = json.load(json_file)
                for single_pedestrian in gt_file:
                    grid_x, grid_y = self.get_worldgrid_from_pos(single_pedestrian['positionID'])
                    # frame_number, personID, x_worldgrid, y_worldgird 
                    gt_out.append([frame_num, grid_x, grid_y])
                    
        gt_out = np.asarray(gt_out)
        np.savetxt(self.gt_fname, gt_out, '%d')
        
    def final_overlap_pos(self):
        train = self.display_cam_layout(self.train_cam)
        mask_train = self.convex_hull(train)
        
        test = self.display_cam_layout(self.test_cam)
        mask_test = self.convex_hull(test)
        
        final_mask = mask_train & mask_test
        final_mask = final_mask.astype(np.uint8)*255.0
        
        coord = np.array(np.where(final_mask==255.0)).T[:,:2]
        coord = np.unique(coord, axis=0)
        print('overlap coord :', coord.shape)
        pos = []
        for p in coord:
            pos.append(self.get_pos_from_worldgrid(p))
        pos = np.asarray(pos)
        print('overlap pos :', pos.shape)
        return pos
        
    def display_cam_layout(self, cam_selected):
        tmap_final = np.zeros(self.world_grid_shape).astype(int)
        for cam in cam_selected:
            i_s, j_s, v_s = [],[],[]
            for i in range(np.product(self.world_grid_shape)):
                grid_x, grid_y = self.get_worldgrid_from_pos(i)
                if i in self.bbox_by_pos_cam:
                    if self.bbox_by_pos_cam[i][cam] > 0:
                        i_s.append(grid_x)
                        j_s.append(grid_y)
                        v_s.append(1)
            tmap = coo_matrix((v_s, (i_s, j_s)), shape=self.world_grid_shape).toarray()
            
            tmap_final+=tmap
            
            '''
            plt.figure(figsize=(10,10))
            plt.subplot(w.num_cam,1,cam+1)
            plt.title('cam_'+str(cam))
            plt.axis('off')
            #plt.imshow(tmap)
            plt.imshow(tmap)
        
        
        plt.figure(figsize=(10,10))
        plt.title('final')
        plt.axis('off')
        #plt.imshow(tmap)
        plt.imshow(tmap_final, cmap='gray')
        #plt.colorbar(tmap_final)
        plt.show()
        '''
        return tmap_final
    
    def convex_hull(self, tmap):
        tmap = tmap.astype(np.uint8)*255
        '''
        cv2.imshow('ConvexHull', tmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        
        #gray = cv2.cvtColor(tmap, cv2.COLOR_BGR2GRAY) # convert to grayscale
        blur = cv2.blur(tmap, (3, 3)) # blur the image
        ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # create hull array for convex hull points
        hull = []
        
        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))
            
        # create an empty black image
        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
        
        # draw contours and hull points
        for i in range(len(contours)):
            color_contours = (0, 255, 0) # green - color for contours
            color = (255, 0, 0) # blue - color for convex hull
            # draw ith contour
            #cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
            # draw ith convex hull object
            #cv2.drawContours(drawing, hull, i, color, 1, 8)
            cv2.fillPoly(drawing , contours, (255, 255, 255))
            
        mask = drawing == 255
        return mask
    
    def read_POM2(self):
        filename = 'rectangles.pom'
        bbox_by_pos_cam = {}
        cam_pos_pattern = re.compile(r'(\d+) (\d+)')
        cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
        with open(os.path.join(self.root,filename),'r') as f:
            for line in f:
                if 'RECTANGLE' in line:
                    cam, pos = map(int,cam_pos_pattern.search(line).groups())
                    if cam != 9 :
                        if pos not in bbox_by_pos_cam:
                            bbox_by_pos_cam[pos] = {}
                            #bbox_by_pos_cam[pos] = 0
                        if 'notvisible' in line:
                            bbox_by_pos_cam[pos][cam] = 0
                            #pass
                        else:
                            bbox_by_pos_cam[pos][cam] = 1  
                        #cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                        #grid_x, grid_y = self.get_worldgrid_from_pos(pos)
                        #bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0), min(right, 1920 - 1), min(bottom, 1080 - 1)]
                        #bbox_by_pos_cam[pos][cam] = [grid_x, grid_y]
                        
        return bbox_by_pos_cam
