import tensorflow as tf
import os
import numpy as np
import scipy.misc
import argparse
from model_ import cycleGAN

#============================================================================================

parser = argparse.ArgumentParser(description='')
#Training Setting
parser.add_argument('--input_width_x', dest='input_width_x', default=128, help='input_x width')
parser.add_argument('--input_height_x', dest='input_height_x', default=128, help='input_x height')
parser.add_argument('--input_width_y', dest='input_width_y', default=128, help='input_y width')
parser.add_argument('--input_height_y', dest='input_height_y', default=128, help='input_y height')
parser.add_argument('--batch_size', dest='batch_size', default=1, help='batch size')
parser.add_argument('--sample_size', dest='sample_size', default=64, help='sample size')
parser.add_argument('--channel_dim', dest='channel_dim', default=3, help='channel dimension')

parser.add_argument('--data_path', dest='data_path', default='./datasets', help='training datapath')
parser.add_argument('--data', dest='data', default='apple2orange', help='data')
parser.add_argument('--trainsetA', dest='trainsetA', default='trainA', help='A training folder name')
parser.add_argument('--trainsetB', dest='trainsetB', default='trainB', help='B training folder name')
parser.add_argument('--image_type', dest='image_type', default='*.jpg', help='training image data type')

parser.add_argument('--logpoint', dest='logpoint', default=1000, help='steps to make save')
parser.add_argument('--log_folder', dest='log_folder', default='./checkpoints', help='folder to make saves')
parser.add_argument('--log_models', dest='log_models', default='models', help='model folder name')
parser.add_argument('--log_images', dest='log_images', default='images', help='image folder name')
parser.add_argument('--graphs', dest='graphs', default='./graphs', help='graph folder')
#Network Settings
parser.add_argument('--max_epochs', dest='max_epochs', default=25, help='max number of epochs')
parser.add_argument('--G_AB_iter', dest='G_AB_iter', default=1, help='iteration of G_AB')
parser.add_argument('--G_BA_iter', dest='G_BA_iter', default=1, help='iteration of G_BA')
parser.add_argument('--Dx_iter', dest='Dx_iter', default=1, help='iteration of D_x')
parser.add_argument('--Dy_iter', dest='Dy_iter', default=1, help='iteration of D_y')
parser.add_argument('--lambda_val', dest='lambda_val', default=10, help='lambda value')

parser.add_argument('--learning_rate', dest='learning_rate', default=0.0002, help='learning rate of the model')
parser.add_argument('--momentum', dest='momentum', default=0.5, help='momentum of the model')
parser.add_argument('--resnet_size', dest='resnet_size', default=9, help='# of residual block')

#Test settings
parser.add_argument('--is_train', dest='is_train', default=False, help='training trigger')
parser.add_argument('--test_A', dest='test_A', default='testA', help='A test folder name')
parser.add_argument('--test_B', dest='test_B', default='testB', help='B test folder name')
parser.add_argument('--A_path', dest='A_path', default='res_A', help='test folder path name')
parser.add_argument('--B_path', dest='B_path', default='res_B', help='test folder path name')
parser.add_argument('--fake_A_path', dest='fake_A_path', default='fake_A', help='test folder path name')
parser.add_argument('--fake_B_path', dest='fake_B_path', default='fake_B', help='test folder path name')
parser.add_argument('--test_image_type', dest='test_image_type', default='.jpg', help='test image data type')

args = parser.parse_args()
#============================================================================================

def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = False
    print 'Starting CycleGAN'
    with tf.Session(config=run_config) as sess:
        
        model = cycleGAN(sess, args)
        if args.is_train:
            print 'Created CycleGAN Network...'
            print 'Start Training...'
            model.train(args)
        else:
            print 'Generating Images...'
            model.make_data(args)

main(args)

#Still Working....