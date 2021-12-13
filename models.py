#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join;
import wget;
import tensorflow as tf;

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
