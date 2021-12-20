#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join;
import wget;
import numpy as np;
import tensorflow as tf;
import tensorflow_addons as tfa;

def PoseResNet(img_shape = (608, 608), num_resnet_layers = 18, use_fpn = True, head_hidden_channels = 64, heads_output_channels = {'hm_cen': 3, 'cen_offset': 2, 'direction': 2, 'z_coor': 1, 'dim': 3}, **kwargs):
  assert type(img_shape) in [list, tuple] and len(img_shape) == 2;
  assert type(num_resnet_layers) is int and num_resnet_layers in [18, 34, 50, 101, 152];
  assert head_hidden_channels is None or type(head_hidden_channels) is int;
  if num_resnet_layers == 18:
    if not exists('models'): mkdir('models');
    if not exists(join('models', 'resnet18.h5')): wget.download('https://github.com/breadbread1984/resnet18-34/raw/master/models/resnet18.h5', out = 'models');
    resnet = tf.keras.models.load_model(join('models', 'resnet18.h5'));
    inputs = tf.keras.Input((None, None, 3));
    outputs = list();
    results = inputs;
    for i in range(1, len(resnet.layers)):
      results = resnet.layers[i](results);
      if resnet.layers[i].name in ['model_1', 'model_3', 'model_5', 'model_7']:
        outputs.append(results);
    output1, output2, output3, output4 = tuple(outputs);
  elif num_resnet_layers == 34:
    if not exists('models'): mkdir('models');
    if not exists(join('models', 'resnet34.h5')): wget.download('https://github.com/breadbread1984/resnet18-34/raw/master/models/resnet34.h5', out = 'models');
    resnet = tf.keras.models.load_model(join('models', 'resnet34.h5'));
    inputs = tf.keras.Input((None, None, 3));
    outputs = list();
    results = inputs;
    for i in range(1, len(resnet.layers)):
      results = resnet.layers[i](results);
      if resnet.layers[i].name in ['model_2', 'model_6', 'model_12', 'model_15']:
        outputs.append(results);
    output1, output2, output3, output4 = tuple(outputs);
  elif num_resnet_layers == 50:
    resnet = tf.keras.applications.ResNet50(input_shape = (img_shape[0], img_shape[1], 3), include_top = False, weights = 'imagenet');
    inputs = resnet.input;
    output1 = resnet.get_layer('conv2_block3_out').output;
    output2 = resnet.get_layer('conv3_block4_out').output;
    output3 = resnet.get_layer('conv4_block6_out').output;
    output4 = resnet.get_layer('conv5_block3_out').output;
  elif num_resnet_layers == 101:
    resnet = tf.keras.applications.resnet.ResNet101(input_shape = (img_shape[0], img_shape[1], 3), include_top = False, weights = 'imagenet');
    inputs = resnet.input;
    output1 = resnet.get_layer('conv2_block3_out').output;
    output2 = resnet.get_layer('conv3_block4_out').output;
    output3 = resnet.get_layer('conv4_block23_out').output;
    output4 = resnet.get_layer('conv5_block3_out').output;
  elif num_resnet_layers == 152:
    resnet = tf.keras.applications.resnet.ResNet152(input_shape = (img_shape[0], img_shape[1], 3), include_top = False, weights = 'imagenet');
    inputs = resnet.input;
    output1 = resnet.get_layer('conv2_block3_out').output;
    output2 = resnet.get_layer('conv3_block8_out').output;
    output3 = resnet.get_layer('conv4_block36_out').output;
    output4 = resnet.get_layer('conv5_block3_out').output;
  else:
    raise Exception('unknown configuration!');
  model = tf.keras.Model(inputs = inputs, outputs = [output1, output2, output3, output4]);
  inputs = tf.keras.Input((img_shape[0], img_shape[1], 3)); # inputs.shape = (batch, h, w, c)
  out_layer1, out_layer2, out_layer3, out_layer4 = model(inputs); # results.shape = (batch, h/32, w/32, 2048)
  if use_fpn:
    upsample_level1 = tf.keras.layers.UpSampling2D(size = (2,2), interpolation = 'bilinear')(out_layer4); # upsample_level1.shape = (batch, h/16, w/16, 2048)
    concat_level1 = tf.keras.layers.Concatenate()([upsample_level1, out_layer3]);
    results = tf.keras.layers.Conv2D(256, (1,1), strides = (1,1), padding = 'same')(concat_level1);
    upsample_level2 = tf.keras.layers.UpSampling2D(size = (2,2), interpolation = 'bilinear')(results); # upsample_level2.shape = (batch, h/8, w/8, 256)
    concat_level2 = tf.keras.layers.Concatenate()([upsample_level2, out_layer2]);
    results = tf.keras.layers.Conv2D(128, (1,1), strides = (1,1), padding = 'same')(concat_level2);
    upsample_level3 = tf.keras.layers.UpSampling2D(size = (2,2), interpolation = 'bilinear')(results); # upsample_level3.shape = (batch, h/4, w/4, 128)
    concat_level3 = tf.keras.layers.Concatenate()([upsample_level3, out_layer1]);
    upsample_level4 = tf.keras.layers.Conv2D(64, (1,1), strides = (1,1), padding = 'same')(concat_level3); # upsample_level4.shape = (batch, h/4, w/4, 64)
    outputs = list();
    if head_hidden_channels is not None:
      for head_name, channels in heads_output_channels.items():
        results = tf.keras.layers.Conv2D(head_hidden_channels, (3,3), padding = 'same', activation = tf.keras.activations.relu)(upsample_level2);
        results_level2 = tf.keras.layers.Conv2D(channels, (1,1), padding = 'same')(results); # results_level2.shape = (batch, h/8, w/8, channels)
        results_level2 = tf.keras.layers.UpSampling2D(size = (2,2), interpolation = 'nearest')(results_level2); # results_level2.shape = (batch, h/4, w/4, channels)
        results = tf.keras.layers.Conv2D(head_hidden_channels, (3,3), padding = 'same', activation = tf.keras.activations.relu)(upsample_level3);
        results_level3 = tf.keras.layers.Conv2D(channels, (1,1), padding = 'same')(results); # results_level3.shape = (batch, h/4, w/4, channels)
        results = tf.keras.layers.Conv2D(head_hidden_channels, (3,3), padding = 'same', activation = tf.keras.activations.relu)(upsample_level4);
        results_level4 = tf.keras.layers.Conv2D(channels, (1,1), padding = 'same')(results); # results_level4.shape = (batch, h/4, w/4, channels)
        results = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis = -1))([results_level2, results_level3, results_level4]); # results.shape = (batch, h/4, w/4, channels, 3)
        weights = tf.keras.layers.Softmax(axis = -1)(results); # weights.shape = (batch, h/4, w/4, channels, 3)
        results = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = -1), name = head_name)([results, weights]); # results.shape = (batch, h/4, w/4, channels)
        outputs.append(results);
    else:
      for head_name, channels in heads_output_channels.items():
        results_level2 = tf.keras.layers.Conv2D(channels, (1,1), padding = 'same')(upsample_level2); # results_level2.shape = (batch, h/8, w/8, channels)
        results_level2 = tf.keras.layers.UpSampling2D(size = (2,2), interpolation = 'nearest')(results_level2); # results_level2.shape = (batch, h/4, w/4, channels)
        results_level3 = tf.keras.layers.Conv2D(channels, (1,1), padding = 'same')(upsample_level3); # results_level3.shape = (batch, h/4, w/4, channels)
        results_level4 = tf.keras.layers.Conv2D(channels, (1,1), padding = 'same')(upsample_level4); # results_level4.shape = (batch, h/4, w/4, channels)
        results = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis = -1))([results_level2, results_level3, results_level4]); # results.shape = (batch, h/4, w/4, channels, 3)
        weights = tf.keras.layers.Softmax(axis = -1)(results); # weights.shape = (batch, h/4, w/4, channels, 3)
        results = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = -1), name = head_name)([results, weights]); # results.shape = (batch, h/4, w/4, channels)
        outputs.append(results);
  else:
    results = out_layer4;
    for i in range(3):
      results = tf.keras.layers.Conv2DTranspose(256, (4,4), strides = (2,2), padding = 'same', use_bias = False)(results);
      results = tf.keras.layers.BatchNormalization()(results);
      results = tf.keras.layers.ReLU()(results);
    # NOTE: results.shape = (batch, h /4, w/4, 256)
    outputs = list();
    if head_hidden_channels is not None:
      for head_name, channels in heads_output_channels.items():
        model = tf.keras.Sequential();
        model.add(tf.keras.layers.Conv2D(head_hidden_channels, (3,3), padding = 'same', activation = tf.keras.activations.relu));
        model.add(tf.keras.layers.Conv2D(channels, (1,1), padding = 'same', name = head_name));
        outputs.append(model(results));
    else:
      for head_name, channels in heads_output_channels.items():
        outputs.append(tf.keras.layers.Conv2D(channels, (1,1), padding = 'same', name = head_name)(results));
  return tf.keras.Model(inputs = inputs, outputs = outputs, **kwargs);

def Gather3D(axis):
  assert type(axis) is int and 0 <= axis <= 2;
  inputs = tf.keras.Input((None, None));
  ind = tf.keras.Input((None, None), dtype = tf.int32);
  x = tf.keras.layers.Lambda(lambda x: tf.cast(tf.tile(tf.reshape(tf.range(tf.shape(x)[0]), (-1, 1, 1)), (1, tf.shape(x)[1], tf.shape(x)[2])), dtype = tf.int32))(ind); # x.shape = ind_shape
  y = tf.keras.layers.Lambda(lambda x: tf.cast(tf.tile(tf.reshape(tf.range(tf.shape(x)[1]), (1, -1, 1)), (tf.shape(x)[0], 1, tf.shape(x)[2])), dtype = tf.int32))(ind); # y.shape = ind_shape
  z = tf.keras.layers.Lambda(lambda x: tf.cast(tf.tile(tf.reshape(tf.range(tf.shape(x)[2]), (1, 1, -1)), (tf.shape(x)[0], tf.shape(x)[1], 1)), dtype = tf.int32))(ind); # z.shape = ind_shape
  if axis == 0:
    coord = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis = -1))([ind, y, z]); # ind_shape x 3
  elif axis == 1:
    coord = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis = -1))([x, ind, z]); # ind_shape x 3
  elif axis == 2:
    coord = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis = -1))([x, y, ind]); # ind_shape x 3
  else:
    raise Exception('invalid axis');
  results = tf.keras.layers.Lambda(lambda x: tf.gather_nd(x[0], x[1]))([inputs, coord]); # results.shape = ind_shape
  return tf.keras.Model(inputs = (inputs, ind), outputs = results);

def L1Loss(channels, max_objects = 50, balanced = False, alpha = 0.5, beta = 1., gamma = 1.5):
  pred = tf.keras.Input((None, None, channels)); # pred.shape = (batch, hm_size, hm_size, channels)
  gt = tf.keras.Input((max_objects, channels)); # gt.shape = (batch, max_objects, channels)
  indices_center = tf.keras.Input([max_objects], dtype = tf.int32); # indices_center.shape = (batch, max_objects)
  obj_mask = tf.keras.Input([max_objects]); # obj_mask.shape = (batch, max_objects)
  # only focus on locations having objects
  feat = tf.keras.layers.Reshape((-1, channels))(pred); # feat.shape = (batch, hm_size * hm_size, channels)
  ind = tf.keras.layers.Lambda(lambda x,c: tf.tile(tf.expand_dims(x, axis = -1), (1,1,c)), arguments = {'c': channels})(indices_center); # ind.shape = (batch, max_objects, channels)
  feat = Gather3D(axis = 1)([feat, ind]); # feat.shape = (batch, max_objects, channels)
  mask = tf.keras.layers.Lambda(lambda x,c: tf.tile(tf.expand_dims(x, axis = -1), (1,1,c)), arguments = {'c': channels})(obj_mask); # mask.shape = (batch, max_objects, channels)
  masked_feat = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[0] * x[1], axis = -1))([feat, mask]); # masked_feat.shape = (batch, max_objects, channels, 1)
  masked_gt = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[0] * x[1], axis = -1))([gt, mask]); # masked_gt.shape = (batch, max_objects, channels, 1)
  if balanced == True:
    diff = tf.keras.layers.Lambda(lambda x: tf.math.abs(x[0] - x[1]))([masked_gt, masked_feat]); # diff.shape = (batch, max_objects, channels, 1)
    loss = tf.keras.layers.Lambda(lambda x,a,b,c,d: tf.math.reduce_sum(
      tf.where(
        tf.math.less(x, b),
        a / d * (d * x + 1) * tf.math.log(d * x / b + 1) - a * x,
        c * x + c / d - a * b
      )), arguments = {'a': alpha, 'b': beta, 'c': gamma, 'd': np.exp(gamma/alpha) - 1})(diff);
  else:
    loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits = True, reduction = tf.keras.losses.Reduction.SUM)(masked_gt, masked_feat);
  loss = tf.keras.layers.Lambda(lambda x: x[0] / (tf.math.reduce_sum(x[1]) + 1e-4))([loss, mask]);
  return tf.keras.Model(inputs = (pred, gt, indices_center, obj_mask), outputs = loss);

def Loss(hm_size = (152, 152), num_classes = 3, max_objects = 50,):
  pred_hm_cen = tf.keras.Input([hm_size[0], hm_size[1], num_classes]);
  pred_cen_offset = tf.keras.Input([hm_size[0], hm_size[1], 2]);
  pred_direction = tf.keras.Input([hm_size[0], hm_size[1], 2]);
  pred_z_coor = tf.keras.Input([hm_size[0], hm_size[1], 1]);
  pred_dim = tf.keras.Input([hm_size[0], hm_size[1], 3]);
  
  hm_cen = tf.keras.Input([num_classes, hm_size[0], hm_size[1]]);
  cen_offset = tf.keras.Input([max_objects, 2]);
  direction = tf.keras.Input([max_objects, 2]);
  z_coor = tf.keras.Input([max_objects, 1]);
  dim = tf.keras.Input([max_objects, 3]);
  
  indices_center = tf.keras.Input([max_objects], dtype = tf.int32);
  obj_mask = tf.keras.Input([max_objects]);
  # 1) hm_cen loss
  gt = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1))(hm_cen); # gt.shape = (num_classes, hm_size, hm_size, 1)
  pred = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1))(pred_hm_cen); # pred.shape = (num_classes, hm_size, hm_size, 1)
  hm_cen_loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits = True, reduction = tf.keras.losses.Reduction.AUTO)(gt, pred); # hm_cen_loss.shape = ()
  # 2) cen_offset loss
  cen_offset_loss = L1Loss(2, max_objects)([pred_cen_offset, cen_offset, indices_center, obj_mask]); # cen_offset_loss.shape = ()
  # 3) direction loss
  direction_loss = L1Loss(2, max_objects)([pred_direction, direction, indices_center, obj_mask]); # direction_loss.shape = ()
  # 4) z_coor loss
  z_coor_loss = L1Loss(1, max_objects, balanced = True)([pred_z_coor, z_coor, indices_center, obj_mask]); # z_coor_loss.shape = ()
  # 5) dim loss
  dim_loss = L1Loss(3, max_objects, balanced = True)([pred_dim, dim, indices_center, obj_mask]); # dim_loss.shape = ()
  
  loss = tf.keras.layers.Add()([hm_cen_loss, cen_offset_loss, direction_loss, z_coor_loss, dim_loss]); # loss.shape = ()
  return tf.keras.Model(inputs = (pred_hm_cen, pred_cen_offset, pred_direction, pred_z_coor, pred_dim, hm_cen, cen_offset, direction, z_coor, dim, indices_center, obj_mask), outputs = loss);

def Trainer(use_fpn = True):
  bev_map = tf.keras.Input((None, None, 3)); # bev_map.shape = (608, 608, 3)
  
  hm_cen = tf.keras.Input([num_classes, hm_size[0], hm_size[1]]);
  cen_offset = tf.keras.Input([max_objects, 2]);
  direction = tf.keras.Input([max_objects, 2]);
  z_coor = tf.keras.Input([max_objects, 1]);
  dim = tf.keras.Input([max_objects, 3]);
  
  indices_center = tf.keras.Input([max_objects], dtype = tf.int32);
  obj_mask = tf.keras.Input([max_objects]);
  
  pred_hm_cen, pred_cen_offset, pred_direction, pred_z_coor, pred_dim = PoseResNet(use_fpn = use_fpn, name = 'sfa3d')(bev_map);
  loss = Loss()([pred_hm_cen, pred_cen_offset, pred_direction, pred_z_coor, pred_dim, hm_cen, cen_offset, direction, z_coor, dim, indices_center, obj_mask]);
  return tf.keras.Model(inputs = (bev_map, hm_cen, cen_offset, direction, z_coor, dim, indices_center, obj_mask), outputs = loss);

if __name__ == "__main__":
  import numpy as np;
  inputs = np.random.normal(size = (4, 608, 608, 3));
  poseresnet = PoseResNet(use_fpn = False, head_hidden_channels = None, name = 'poseresnet1');
  outputs = poseresnet(inputs)
  print([o.shape for o in outputs]);
  poseresnet = PoseResNet(use_fpn = False, head_hidden_channels = 64, name = 'poseresnet2');
  outputs = poseresnet(inputs)
  print([o.shape for o in outputs]);
  poseresnet.save('poseresnet.h5');
  tf.keras.utils.plot_model(poseresnet, to_file = 'poseresnet.png', expand_nested = True);
  poseresnet = PoseResNet(use_fpn = True, head_hidden_channels = None, name = 'poseresnet3');
  outputs = poseresnet(inputs)
  print([o.shape for o in outputs]);
  poseresnet = PoseResNet(use_fpn = True, head_hidden_channels = 64, name = 'poseresnet4');
  outputs = poseresnet(inputs)
  print([o.shape for o in outputs]);
  poseresnet.save('poseresnet_fpn.h5');
  tf.keras.utils.plot_model(poseresnet, to_file = 'poseresnet_fpn.png', expand_nested = True);
