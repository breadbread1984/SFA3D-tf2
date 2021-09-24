#!/usr/bin/python3

from os.path import join;
import numpy as np;
from scipy.optimize import fsolve;
import cv2;
import tensorflow as tf;

class KittiDataset(object):
  CLASS_NAME_TO_ID = {
    'Pedestrian': 0,
    'Car': 1,
    'Cyclist': 2,
    'Van': 1,
    'Truck': -3,
    'Person_sitting': 0,
    'Tram': -99,
    'Misc': -99,
    'DontCare': -1
  };
  Tr_velo_to_cam = np.array([
    [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
    [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
    [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
    [0, 0, 0, 1]
  ]);
  R0 = np.array([
    [0.99992475, 0.00975976, -0.00734152, 0],
    [-0.0097913, 0.99994262, -0.00430371, 0],
    [0.00729911, 0.0043753, 0.99996319, 0],
    [0, 0, 0, 1]
  ]);
  Tr_velo_to_cam_inv = np.linalg.inv(np.array([
    [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
    [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
    [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
    [0, 0, 0, 1]
  ]));
  R0_inv = np.linalg.inv(np.array([
    [0.99992475, 0.00975976, -0.00734152, 0],
    [-0.0097913, 0.99994262, -0.00430371, 0],
    [0.00729911, 0.0043753, 0.99996319, 0],
    [0, 0, 0, 1]
  ]));
  boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
  };
  def __init__(self, data_dir, hm_size, num_classes, max_objects, input_shape = (608, 608)):
    assert type(data_dir) is str;
    assert type(hm_size) is tuple and len(hm_size) == 2;
    assert type(num_classes) is int;
    self.data_dir = data_dir;
    self.hm_size = hm_size; # treat h l regression as classification. how many classes for h and how many classes for l
    self.num_classes = num_classes;
    self.max_objects = max_objects;
    self.input_shape = input_shape;
  def generator(self, mode = 'train'):
    assert mode in ['train', 'test', 'val'];
    sub_folder = 'training' if mode in ['train', 'val'] else 'testing';
    image_dir = join(self.data_dir, sub_folder, 'image_2');
    lidar_dir = join(self.data_dir, sub_folder, 'velodyne');
    calib_dir = join(self.data_dir, sub_folder, 'valib');
    label_dir = join(self.data_dir, sub_folder, 'label_2');
    split_txt_path = join(self.data_dir, 'ImageSets', '%s.txt' % mode);
    sample_id_list = [int(x.strip()) for x in open(split_txt_path).readlines()];
    if mode in ['train', 'val']:
      # load image with labels
      def gen():
        for sample_id in sample_id_list:
          img_path = join(image_dir, "%d.png" % sample_id);
          lidar_path = join(lidar_dir, "%d.bin" % sample_id);
          calib_path = join(calib_dir, "%d.txt" % sample_id);
          label_path = join(label_dir, "%d.txt" % sample_id);
          # 1) TODO: image processing
          # 2) load lidar data
          lidar_data = np.fromfile(lidar_path, dtype = np.float32).reshape(-1,4);
          # 3) load calib data
          with open(calib_path) as f:
            lines = f.readlines();
          p2 = np.array(lines[2].strip().split(' ')[1:], dtype = np.float32).reshape([3, 4]); # intrinsic matrix
          p3 = np.array(lines[3].strip().split(' ')[1:], dtype = np.float32).reshape([3, 4]); # extrinsic matrix
          r0 = np.array(lines[4].strip().split(' ')[1:], dtype = np.float32).reshape([3, 3]); # rotation matrix from RGB camera coord to velodyn camera coord
          v2c = np.array(lines[5].strip().split(' ')[1:], dtype = np.float32).reshape([3, 4]); # velodyn coordinate to camera coordinate transform matrix
          r0_ext = np.eye(4); r0_ext[:3,:3] = r0;
          # v2c = [R|t], c2v = [R'|-R'*t]
          c2v = np.zeros_like(v2c); c2v[:3,:3] = v2c[:3,:3].transpose(); c2v[:3,3] = np.dot(-v2c[:3,:3].transpose(),v2c[:3,3]);
          # 4) load label (object boxes)
          labels = np.zeros((0,8), dtype = np.float32); # labels.shape = (0, 8)
          for line in open(label_path, 'r'):
            line_parts = line.strip().split(' ');
            cat_id = self.CLASS_NAME_TO_ID[line_parts[0]];
            if cat_id <= -99: continue;
            truncated = int(float(line_parts[1])); # truncated pixel ratio [0..1]
            occluded = int(line_parts[2]); # 0=visible,1=partly occluded, 2=fully occluded, 3=unknown
            alpha = float(line_parts[3]); # object observation angle [-pi..pi]
            # xmin, ymin, xmax, ymax
            bbox = np.array([float(line_parts[4]), float(line_parts[5]), float(line_parts[6]), float(line_parts[7])]);
            # box dimension height, width, length (h, w, l)
            h, w, l = float(line_parts[8]), float(line_parts[9]), float(line_parts[10]);
            # center (x,y,z) in camera coord.
            x, y, z = float(line_parts[11]), float(line_parts[12]), float(line_parts[13]);
            # convert location in camera coord to velodyn coord
            p = np.array([x,y,z,1]);
            p = np.matmul(np.linalg.inv(r0_ext), p); # rectified camera coord to RGB camera coord
            p = np.matmul(c2v, p); # RGB camera coord to velodyn coord
            x, y, z = tuple(p[0:3].tolist());
            rz = float(line_parts[14]); # yaw angle [-pi..pi]
            ry = -rz - np.pi / 2;
            # NOTE: object_label = (object category, center x,y,z, box h, w, l, yaw relative to lidar coord)
            object_label = np.array([cat_id, x, y, z, h, w, l, ry], dtype = np.float32);
            labels = np.concatenate([labels, np.expand_dims(object_label, axis = 0)], axis = 0); # labels.shape = (N, 8)
          # 5) augmentation
          if np.random.uniform() < 0.66:
            if np.random.randint(low = 0, high = 2) == 0:
              # random rotation in range [-pi/4, pi/4]
              angle = np.random.uniform(low = -np.pi/4, high = np.pi/4);
              rot = np.eyes(4);
              rot[0,0] = np.cos(angle); rot[0,1] = -np.sin(angle);
              rot[1,0] = np.sin(angle); rot[1,1] = np.cos(angle);
              # 1) point cloud rotation
              points = np.hstack([lidar_data[:,0:3],np.ones((lidar_data[0:3].shape[0],1))]);
              lidar_data[:,0:3] = np.matmul(points, rot)[:, 0:3];
              # 2) boxes rotation
              for object_label in labels:
                # object_label.shape = (8,) in sequence of cls_id,x,y,z,h,w,l,ry
                # 2.1) convert center coordinate and box dimension to box corner coordinates in RGB camera coord
                translation = object_label[1:4]; # translation.shape = (3,)
                h,w,l = tuple(object_label[4:7].tolist()); # size.shape = (3,)
                yaw = object_label[-1]; # yaw.shape = (,)
                # NOTE: trackleBox is 3d coordinates of eight corners
                trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
                  [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
                  [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
                  [0, 0, 0, 0, h, h, h, h]]); # trackletBox.shape = (3,8)
                rotMat = np.array([
                  [np.cos(yaw), -np.sin(yaw), 0.0],
                  [np.sin(yaw), np.cos(yaw), 0.0],
                  [0.0, 0.0, 1.0]]); # rotMat.shape = (3, 3)
                cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T; # cornerPosInVelo.shape = (3,8)
                box3d = cornerPosInVelo.transpose(); # box3d.shape = (8,3)
                # 2.2) rotate eight corners
                points = np.hstack([box3d, np.ones((box3d.shape[0],1))]); # points.shape = (8,4)
                box3d = np.matmul(points, rot)[:, 0:3]; # points.shape = (8,3)
                # 2.3) convert box corner coordinates back to center coordinate and box dimension
                points = np.hstack([box3d, np.ones((box3d.shape[0],1))]).T; # points.shape = (8,4)
                points = np.matmul(self.Tr_velo_to_cam, points); # points.shape = (4,8)
                roi = np.matmul(self.R0, points).T[:,0:3]; # points.shape = (8,3)
                h = abs(np.sum(roi[:4, 1] - roi[4:, 1]) / 4);
                w = np.sum(
                  np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]]) ** 2)) +
                  np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]]) ** 2)) +
                  np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]]) ** 2)) +
                  np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]]) ** 2))
                ) / 4;
                l = np.sum(
                  np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]]) ** 2)) +
                  np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]]) ** 2)) +
                  np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]]) ** 2)) +
                  np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]]) ** 2))
                ) / 4;
                x = np.sum(roi[:, 0], axis=0) / 8;
                y = np.sum(roi[0:4, 1], axis=0) / 4;
                z = np.sum(roi[:, 2], axis=0) / 8;
                ry = np.sum(
                  math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                  math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                  math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                  math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                  math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                  math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                  math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                  math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
                ) / 8;
                if w > l:
                  w, l = l, w;
                  ry = ry - np.pi / 2;
                elif l > w:
                  l, w = w, l;
                  ry = ry - np.pi / 2;
                # 2.4) from RGB camera coord to velodyn camera coord
                p = np.array([x,y,z,1]);
                p = np.matmul(self.R0_inv, p);
                p = np.matmul(self.Tr_velo_to_cam_inv, p)[0:3];
                rz = -ry - np.pi / 2;
                object_label[1:4] = p;
                object_label[4:7] = np.array([h,w,l]);
                object_label[7] = rz;
            else:
              # random scaling in range [0.95, 1.05]
              factor = np.random.uniform(0.95, 1.05);
              lidar_data[:, 0:3] = lidar_data[:, 0:3] * factor; # scale x,y,z
              labels[:, 1:7] = labels[:, 1:7] * factor; # scale x,y,z,h,w,l
            # 3) filtered lidar_data and labels outside boundary
            mask = np.where((lidar_data[:, 0] >= self.boundary['minX']) & (lidar_data[:, 0] <= self.boundary['maxX']) &
                            (lidar_data[:, 1] >= self.boundary['minY']) & (lidar_data[:, 1] <= self.boundary['maxY']) &
                            (lidar_data[:, 2] >= self.boundary['minZ']) & (lidar_data[:, 2] <= self.boundary['maxZ']));
            lidar_data = lidar_data[mask]; # lidar_data.shape = (filtered point num, 3)
            lidar_data[:,2] = lidar_data[:,2] - self.boundary['minZ'];
            label_x = (labels[:, 1] >= self.boundary['minX']) & (labels[:, 1] < self.boundary['maxX']);
            label_y = (labels[:, 2] >= self.boundary['minY']) & (labels[:, 2] < self.boundary['maxY']);
            label_z = (labels[:, 3] >= self.boundary['minZ']) & (labels[:, 3] < self.boundary['maxZ']);
            mask_label = label_x & label_y & label_z;
            labels = labels[mask_label]; # labels.shape = (filtered target num, 8)
            # 4) make bev map from lidar_data
            height = self.input_shape[0] + 1;
            width = self.input_shape[1] + 1;
            pointcloud = np.copy(lidar_data);
            discretization = (self.boundary["maxX"] - self.boundary["minX"]) / self.input_shape[0];
            pointcloud[:, 0] = np.int_(np.floor(pointcloud[:,0] / discretization));
            pointcloud[:, 1] = np.int_(np.floor(pointcloud[:,1] / discretization) + width / 2);
            # sort points according to x,y,z values, z from large to small
            sorted_indices = np.lexsort((-pointcloud[:,2], pointcloud[:,1], pointcloud[:,0]));
            pointcloud = pointcloud[sorted_indices];
            # only left unique x,y with largest z
            _, unique_indices, unique_counts = np.unique(pointcloud[:, 0:2], axis = 0, return_index = True, return_counts = True);
            pointcloud_top = pointcloud[unique_indices];
            heightMap = np.zeros((height, width)); # depth map
            intensityMap = np.zeros((height, width)); # 
            densityMap = np.zeros((height, width));
            max_height = float(np.abs(self.boundary['maxZ'] - self.boundary['minZ']));
            # normalize depth map
            heightMap[np.int_(pointcloud_top[:,0]), np.int_(pointcloud_top[:,1])] = pointcloud_top[:,2] / max_height;
            # normalized counts of every unique x,y
            normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64));
            # intensity map
            intensityMap[np.int_(pointcloud_top[:,0]), np.int_(pointcloud_top[:,1])] = pointcloud_top[:,3];
            # normalized unique count map (how many reflections per position)
            densityMap[np.int_(pointcloud_top[:,0]), np.int_(pointcloud_top[:,1])] = normalizedCounts;
            bev_map = np.stack([intensityMap[:self.input_shape[0],:self.input_shape[1]],
                                heightMap[:self.input_shape[0],:self.input_shape[1]],
                                densityMap[:self.input_shape[0],:self.input_shape[1]]], axis = -1); # rgb_map.shape = (height, width, 3)
            # 5) random horizontal flip
            flip = True if np.random.uniform() < 0.5 else False;
            if flip: bev_map = bev_map[:,::-1,:];
            # 6) generate labels
            num_objects = min(len(labels), self.max_objects);
            hm_l, hm_w = self.hm_size;
            hm_main_center = np.zeros((self.num_classes, hm_l, hm_w), dtype = np.float32);
            cen_offset = np.zeros((self.max_objects, 2), dtype = np.float32);
            direction = np.zeros((self.max_objects, 2), dtype = np.float32);
            z_coor = np.zeros((self.max_objects, 1), dtype = np.float32);
            dimension = np.zeros((self.max_objects, 3), dtype = np.float32);
            
            indices_center = np.zeros((self.max_objects), dtype = np.int64);
            obj_mask = np.zeros((self.max_objects), dtype = np.uint8);
            for k in range(num_objects):
              cls_id, x, y, z, h, w, l, yaw = labels[k];
              cls_id = int(cls_id);
              yaw = -yaw;
              # discard invalid label
              if not (self.boundary['minX'] <= x <= self.boundary['maxX'] and \
                      self.boundary['minY'] <= y <= self.boundary['maxY'] and \
                      self.boundary['minZ'] <= z <= self.boundary['maxZ']):
                continue;
              if h <= 0 or w <= 0 or l <= 0: continue;
              # quantilize l and w
              bbox_l = l / (self.boundary['maxX'] - self.boundary['minX']) * hm_l;
              bbox_w = w / (self.boundary['maxY'] - self.boundary['minY']) * hm_w;
              height, width = np.ceil(bbox_l), np.ceil(bbox_w);
              # get radius
              # larger solution of x^2 - (height + width) x + (width * height * (1 - 0.7) / (1 + 0.7)) = 0
              a1 = 1;
              b1 = height + width;
              c1 = width * height * (1 - 0.7) / (1 + 0.7);
              sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1);
              r1 = (b1 + sq1) / 2;
              # larger solution of 4 x^2 - 2 * (height + width) x + (1 - 0.7) * width * height = 0
              a2 = 4;
              b2 = 2 * (height + width);
              c2 = (1 - 0.7) * width * height;
              sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2);
              r2 = (b2 + sq2) / 2;
              # larger solution of 4 * 0.7 x^2 + 2 * 0.7 * (height + width) x + (0.7 - 1) * width * height = 0
              a3 = 4 * 0.7;
              b3 = -2 * 0.7 * (height + width);
              c3 = (0.7 - 1) * width * height;
              sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3);
              r3 = (b3 + sq3) / 2;
              radius = max(0, int(min(r1, r2, r3)));
              
          yield lidar_data, labels;
    else:
      def gen():
        
      # load image only
    return gen;
