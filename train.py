#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
from absl import flags, app;
import numpy as np;
import tensorflow as tf;
import tensorflow_addons as tfa;
from create_dataset import KittiDataset;
from models import Trainer;

FLAGS = flags.FLAGS;
flags.DEFINE_integer('batch_size', default = 16, help = 'batch size');
flags.DEFINE_float('lr', default = 1e-3, help = 'learning rate');
flags.DEFINE_list('steps', default = ['150', '180'], help = 'decay steps');
flags.DEFINE_integer('epochs', default = 300, help = 'epochs');
flags.DEFINE_string('checkpoint', default = 'checkpoints', help = 'checkpoint path');
flags.DEFINE_boolean('save_model', default = False, help = 'save model from checkpoint');
flags.DEFINE_boolean('use_fpn', default = True, help = 'use feature pyramid network');
flags.DEFINE_string('kitti_path', default = './kitti', help = 'path to kitti dataset');

class SummaryCallbacks(tf.keras.callbacks.Callback):
  def __init__(self, sfa3d, validationset, eval_freq = 500):
    self.sfa3d = sfa3d;
    self.iter = iter(validationset);
    self.log = tf.summary.create_file_writer(FLAGS.checkpoint);
  def on_batch_end(self, batch, logs = None):
    if batch % slf.eval_freq == 0:
      (bev_map, images), _ = next(self.iter);
      pred_hm_cen, pred_cen_offset, pred_direction, pred_z_coor, pred_dim = self.sfa3d(bev_map);
      # TODO: visualize detection results
      with self.log.as_default():
        for key, value in logs.items():
          tf.summary.scalar(key, value, step = self.sfa3d.optimizr.iterations);

def Loss(_, loss):
  return loss;

def main(unused_argv):
  if exists(join(FLAGS.checkpoint, 'ckpt')):
    trainer = tf.keras.models.load_model(join(FLAGS.checkpoint, 'ckpt'), custom_objects = {'tf': tf, 'Loss': Loss}, compile = True);
    optimizer = trainer.optimizer;
    if FLAGS.save_model:
      if not exists('models'): mkdir('models');
      trainer.get_layer('sfa3d').save(join('models', 'sfa.h5'));
      trainer.get_layer('sfa3d').save_weights(join('models', 'sfa_weights.h5'));
      exit();
  else:
    trainer = Trainer(use_fpn = FLAGS.use_fpn);
    optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(FLAGS.lr, decay_steps = np.ceil(6000 / FLAGS.batch_size), alpha = 0.1));
    trainer.compile(optimizer = optimizer, loss = Loss);
  # create dataset
  kitti = KittiDataset(data_dir = FLAGS.kitti_path);
  trainset, testset = kitti_dataset.load_dataset();
  options = tf.data.Options();
  options.autotune.enabled = True;
  trainset = trainset.with_options(options).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size);
  testset = testset.with_options(options).batch(1);
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = FLAGS.checkpoint),
    tf.keras.callbacks.ModelCheckpoint(filepath = join(FLAGS.checkpoint, 'ckpt'), save_freq = 1000),
    SummaryCallbacks(trainer.get_layer('sfa3d'), testset, eval_freq = 500),
  ];
  trainer.fit(trainset, epochs = FLAGS.epochs, callbacks = callbacks);

if __name__ == "__main__":
  app.run(main);
