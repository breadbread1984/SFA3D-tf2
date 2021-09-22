#!/usr/bin/python3

from os.path import join;
import numpy as np;
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
  def __init__(self, data_dir, hm_size, num_classes, max_objects):
    self.data_dir = data_dir;
    self.hm_size = hm_size;
    self.num_classes = num_classes;
    self.max_objects = max_objects;
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
          '''
          P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                      0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                      0,      0,      1,      0]
                   = K * [1|t]
          '''
          p2 = np.array(lines[2].strip().split(' ')[1:], dtype = np.float32).reshape([3, 4]); # intrinsic matrix
          p3 = np.array(lines[3].strip().split(' ')[1:], dtype = np.float32).reshape([3, 4]); # extrinsic matrix
          r0 = np.array(lines[4].strip().split(' ')[1:], dtype = np.float32).reshape([3, 3]); # rotation matrix from reference camera coord to rect camera coord
          v2c = np.array(lines[5].strip().split(' ')[1:], dtype = np.float32).reshape([3, 4]); # velodyn coordinate to camera coordinate transform matrix
          r0_ext = np.eye(4); r0_ext[:3,:3] = r0;
          inv_tr = np.zeros_like(v2c); inv_tr[:3,:3] = v2c[:3,:3].transpose(); inv_tr[:3,3] = np.dot(-v2c[:3,:3].transpose(),v2c[:3,3]);
          # 4) load label
          labels = np.zeros((0,8), dtype = np.float32);
          for line in open(label_path, 'r'):
            line_parts = line.strip().split(' ');
            cat_id = self.CLASS_NAME_TO_ID[line_parts[0]];
            if cat_id <= -99: continue;
            truncated = int(float(line_parts[1])); # truncated pixel ratio [0..1]
            occluded = int(line_parts[2]); # 0=visible,1=partly occluded, 2=fully occluded, 3=unknown
            alpha = float(line_parts[3]); # object observation angle [-pi..pi]
            # xmin, ymin, xmax, ymax
            bbox = np.array([float(line_parts[4]), float(line_parts[5]), float(line_parts[6]), float(line_parts[7])]);
            # height, width, length (h, w, l)
            h, w, l = float(line_parts[8]), float(line_parts[9]), float(line_parts[10]);
            # location (x,y,z) in camera coord.
            x, y, z = float(line_parts[11]), float(line_parts[12]), float(line_parts[13]);
            # convert location in camera coord to lidar box
            p = np.array([x,y,z,1]);
            p = np.matmul(np.linalg.inv(r0_ext), p);
            p = np.matmul(inv_tr, p);
            x, y, z = tuple(p[0:3].tolist());
            rz = float(line_parts[14]); # yaw angle [-pi..pi]
            ry = -rz - np.pi / 2;
            # NOTE: object_label = (object category, center x,y,z, box h, w, l, yaw relative to lidar coord)
            object_label = np.array([cat_id, x, y, z, h, w, l, ry], dtype = np.float32);
            labels = np.concatenate([labels, np.expand_dims(object_label, axis = 0)], axis = 0);
          # 5) augmentation
          if np.random.uniform() < 0.66:
            if np.random.randint(low = 0, high = 2) == 0:
              # random rotation in range [-pi/4, pi/4]
              angle = np.random.uniform(low = -np.pi/4, high = np.pi/4);
              # point transform
              points = np.hstack([lidar_data[:,0:3],np.ones((lidar_data[0:3].shape[0],1))]);
              rot = np.eyes(4);
              rot[0,0] = np.cos(angle); rot[0,1] = -np.sin(angle);
              rot[1,0] = np.sin(angle); rot[1,1] = np.cos(angle);
              lidar_data[:,0:3] = np.matmul(points, rot);
              # box transform
              ret = np.zeros((0,8), dtype = np.float32);
              for object_label in labels:
                translation = object_label[1:4]; # center
                h,w,l = tuple(object_label[4:7].tolist()); # size
                rotation = [0,0, object_label[-1]]; # yaw angle
                trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
                  [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
                  [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
                  [0, 0, 0, 0, h, h, h, h]]);
                yaw = rotation[-1];
                rotMat = np.array([
                  [np.cos(yaw), -np.sin(yaw), 0.0],
                  [np.sin(yaw), np.cos(yaw), 0.0],
                  [0.0, 0.0, 1.0]]);
                cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T;
                box3d = cornerPosInVelo.transpose();
                ret = np.concatenate([ret, np.expand_dims(box3d, axis = 0)], axis = 0);
              labels = ret
            else:
              # random scaling
              
    else:
      # load image only
    return gen;
