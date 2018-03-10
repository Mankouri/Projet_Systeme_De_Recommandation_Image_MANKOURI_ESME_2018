#!/usr/bin/env python

#   MANKOURI Jalil - 3CI
#   $ cd /mnt/c/Users/rushi/Desktop/Projet/tools/
#   $ wget -O /tmp/cat.jpg https://farm6.staticflickr.com/5470/9372235876_d7d69f1790_b.jpg
#   $ python classify.py /tmp/cat.jpg

# Presentation python classify.py /mnt/c/img/*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.python.training import saver as tf_saver

slim = tf.contrib.slim
FLAGS = None

def traitementImage(image, central_fraction=0.875):
  # On prend l'image sous son format JPG puis on la convertie en float pour
  # la suite.
  image = tf.cast(tf.image.decode_jpeg(image, channels=3), tf.float32)
  image = tf.image.central_crop(image, central_fraction=central_fraction)
  image = tf.expand_dims(image, [0])
  image = tf.image.resize_bilinear(image,
                                 [FLAGS.image_size, FLAGS.image_size],
                                 align_corners=False)

  image = tf.multiply(image, 1.0/127.5)
  # On retourne l'image traiter.
  return tf.subtract(image, 1.0)


def chargementDesLabels(num_classes, labelmap_path, dict_path):
    
  labelmap = [line.rstrip() for line in tf.gfile.GFile(labelmap_path).readlines()]
  if len(labelmap) != num_classes:
    tf.logging.fatal(
        "Label map charger {} contient: {} lignes, nombre de classe {}".format(
            labelmap_path, len(labelmap), num_classes))
    sys.exit(1)

  label_dict = {}
  for line in tf.gfile.GFile(dict_path).readlines():
    words = [word.strip(' "\n') for word in line.split(',', 1)]
    label_dict[words[0]] = words[1]

  return labelmap, label_dict


def main(args):
  if not os.path.exists(FLAGS.checkpoint):
    tf.logging.fatal(
        ' %s . executer les commandes shell pour telecharger le modele',
        FLAGS.checkpoint)
  g = tf.Graph()
  with g.as_default():
    input_image = tf.placeholder(tf.string)
    traiter_image = traitementImage(input_image)

    with slim.arg_scope(inception.inception_v3_arg_scope()):
      logits, end_points = inception.inception_v3(
          traiter_image, num_classes=FLAGS.num_classes, is_training=False)

    predictions = end_points['multi_predictions'] = tf.nn.sigmoid(
        logits, name='multi_predictions')
    saver = tf_saver.Saver()
    sess = tf.Session()
    saver.restore(sess, FLAGS.checkpoint)

    # On lance l'evaluation sur les images
    for image_path in FLAGS.image_path:
      if not os.path.exists(image_path):
        tf.logging.fatal('Image entree ne correspond pas:  %s', FLAGS.image_path[0])
      img_data = tf.gfile.FastGFile(image_path, "rb").read()
      print(image_path)
      predictions_eval = np.squeeze(sess.run(predictions,
                                             {input_image: img_data}))

      # On affiche les meilleurs resultats (n) "score"
      labelmap, label_dict = chargementDesLabels(FLAGS.num_classes, FLAGS.labelmap, FLAGS.dict)

      top_k = predictions_eval.argsort()[-FLAGS.n:][::-1]
      for idx in top_k:
        mid = labelmap[idx]
        display_name = label_dict.get(mid, 'unknown')
        score = predictions_eval[idx]
        print('{}: {} - {} (score = {:.2f})'.format(idx, mid, display_name, score))
      print()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', type=str, default='../data/2016_08/model.ckpt',
                      help='Checkpoint to run inference on.')
  parser.add_argument('--labelmap', type=str, default='../data/2016_08/labelmap.txt',
                      help='Label map that translates from index to mid.')
  parser.add_argument('--dict', type=str, default='../dict.csv',
                      help='Path to a dict.csv that translates from mid to a display name.')
  parser.add_argument('--image_size', type=int, default=299,
                      help='Image size to run inference on.')
  parser.add_argument('--num_classes', type=int, default=6012,
                      help='Number of output classes.')
  parser.add_argument('--n', type=int, default=10,
                      help='Number of top predictions to print.')
  parser.add_argument('image_path', nargs='+', default='')
  FLAGS = parser.parse_args()
  tf.app.run()
