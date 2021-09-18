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
          lidar_data = np.fromfile(lidar_file, dtype = np.float32).reshape(-1,4);
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
            ry = float(line_parts[14]); # yaw angle [-pi..pi]
            object_label = np.array([cat_id, x, y, z, h, w, l, ry], dtype = np.float32);
            labels = np.concatenate([labels, np.expand_dims(object_label, axis = 0)], axis = 0);
          # TODO
    else:
      # load image only
    return gen;
