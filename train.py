#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
from absl import flags, app;
import numpy as np;
import tensorflow as tf;
from create_dataset import KittiDataset;
from models import PoseResNet;

FLAGS = flags.FLAGS;
flags.DEFINE_integer('batch_size', default = 16, help = 'batch size');
flags.DEFINE_float('lr', default = 1e-3, help = 'learning rate');
flags.DEFINE_list('steps', default = ['150', '180'], help = 'decay steps');
flags.DEFINE_string('checkpoint', default = 'checkpoints', help = 'checkpoint path');
flags.DEFINE_boolean('save_model', default = False, help = 'save model from checkpoint');
flags.DEFINE_boolean('use_fpn', default = True, help = 'use feature pyramid network');

def main(unused_argv):
  if exists(join(FLAGS.checkpoint, 'ckpt')):
    sfa = tf.keras.models.load_model(join(FLAGS.checkpoint, 'ckpt'), custom_objects = {'tf': tf}, compile = True);
    optimizer = model.optimizer;
    if FLAGS.save_model:
      if not exists('models'): mkdir('models');
      sfa.save(join('models', 'sfa.h5'));
      sfa.save_weights(join('models', 'sfa_weights.h5'));
      exit();
  else:
    sfa = PoseResNet(use_fpn = FLAGS.use_fpn);
    optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(FLAGS.lr, decay_steps = np.ceil(6000 / FLAGS.batch_size), alpha = 0.1));
    sfa.compile(optimizer = optimizer,
                loss = {'hm_cen': ,'cen_offset':, 'direction':, 'z_coor':, 'dim':})
    
