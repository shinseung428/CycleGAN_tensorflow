import tensorflow as tf
import os
import numpy as np
import scipy.misc
import argparse
from model_ import cycleGAN

# configs = tf.app.flags

# #Network Parameters
# configs.DEFINE_integer("input_width_x", 128, "Input image width x [128]")
# configs.DEFINE_integer("input_height_x", 128, "Input image height x [128]")
# configs.DEFINE_integer("input_width_y", 128, "Input image width y [128]")
# configs.DEFINE_integer("input_height_y", 128, "Input image width y [128]")
# configs.DEFINE_integer("batch_size", 1, "Batch size [64]")
# configs.DEFINE_integer("sample_size", 64, "Sample size [64]")
# configs.DEFINE_integer("channel_dim", 1, "Image Channel [3]")
# configs.DEFINE_string("resnet_size", "9_blocks", "Resnet size [9]")

# #Training Parameters
# configs.DEFINE_string("data_path", "./data", "Training datapath")
# configs.DEFINE_string("trainsetA", "trainA", "Training Set A")
# configs.DEFINE_string("trainsetB", "trainB", "Training Set B")
# configs.DEFINE_string("image_type", "*.png", "jpeg image")

# configs.DEFINE_integer("max_epochs", 25, "Max number of epochs [25]")
# configs.DEFINE_integer("AB_G_iter", 1, "Generator_AB []")
# configs.DEFINE_integer("BA_G_iter", 1, "Generator_BA []")
# configs.DEFINE_integer("Dx_iter", 1, "Discriminator_x []")
# configs.DEFINE_integer("Dy_iter", 1, "Discriminator_y []")
# configs.DEFINE_integer("lambda_val", 10, "Lambda value [10]")

# configs.DEFINE_float("learning_rate", 0.0002, "AdamOptim learning rate [0.0002]")
# configs.DEFINE_float("momentum", 0.5, "Momentum [0.5]")

# configs.DEFINE_integer("logpoint", 1000, "log point [100]")
# configs.DEFINE_string("checkpoint", "./checkpoints", "Checkpoint folder [checkpoint]")
# configs.DEFINE_string("checkpoint_model", "models", "Model folder [model]")
# configs.DEFINE_string("checkpoint_images", "images", "Images folder [images]")

# #Generate data using the model
# configs.DEFINE_boolean("is_train", False, "Train model [False]")
# configs.DEFINE_string("test_A", "trainA", "Synthetic images folder [synt_imagesA]")
# configs.DEFINE_string("test_B", "trainB", "Synthetic images folder [synt_imagesB]")
# configs.DEFINE_string("test_path", "res_fake", "Test result path [test_res]")
# configs.DEFINE_string("test_image_type", ".png", "Test image type [png]")

# config = configs.FLAGS
#============================================================================================

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_width_x', dest='input_width_x', default=128, help='input_x width')
parser.add_argument('--input_height_x', dest='input_height_x', default=128, help='input_x height')
parser.add_argument('--input_width_y', dest='input_width_y', default=128, help='input_y width')
parser.add_argument('--input_height_y', dest='input_height_y', default=128, help='input_y height')

parser.add_argument('--batch_size', dest='batch_size', default=1, help='batch size')
parser.add_argument('--sample_size', dest='sample_size', default=64, help='sample size')
parser.add_argument('--channel_dim', dest='channel_dim', default=1, help='channel dimension')
parser.add_argument('--resnet_size', dest='resnet_size', default=9, help='# of residual block')

parser.add_argument('--data_path', dest='data_path', default='./data', help='training datapath')
parser.add_argument('--trainsetA', dest='trainsetA', default='trainA', help='A training folder name')
parser.add_argument('--trainsetB', dest='trainsetB', default='trainB', help='B training folder name')
parser.add_argument('--image_type', dest='image_type', default='*png', help='training image data type')

parser.add_argument('--max_epochs', dest='max_epochs', default=25, help='max number of epochs')
parser.add_argument('--G_AB_iter', dest='G_AB_iter', default=1, help='iteration of G_AB')
parser.add_argument('--G_BA_iter', dest='G_BA_iter', default=1, help='iteration of G_BA')
parser.add_argument('--Dx_iter', dest='Dx_iter', default=1, help='iteration of D_x')
parser.add_argument('--Dy_iter', dest='Dy_iter', default=1, help='iteration of D_y')
parser.add_argument('--lambda_val', dest='lambda_val', default=10, help='lambda value')

parser.add_argument('--learning_rate', dest='learning_rate', default=0.0002, help='learning rate of the model')
parser.add_argument('--momentum', dest='momentum', default=0.5, help='momentum of the model')

parser.add_argument('--logpoint', dest='logpoint', default=1000, help='steps to make save')
parser.add_argument('--log_folder', dest='log_folder', default='./checkpoints', help='folder to make saves')
parser.add_argument('--log_models', dest='log_models', default='models', help='model folder name')
parser.add_argument('--log_images', dest='log_images', default='images', help='image folder name')

parser.add_argument('--is_train', dest='is_train', default=False, help='training trigger')
parser.add_argument('--test_A', dest='test_A', default='trainA', help='A test folder name')
parser.add_argument('--test_B', dest='test_B', default='trainB', help='B test folder name')
parser.add_argument('--A_path', dest='A_path', default='res_A', help='test folder path name')
parser.add_argument('--B_path', dest='B_path', default='res_B', help='test folder path name')
parser.add_argument('--test_path', dest='test_path', default='res_fake', help='test folder path name')
parser.add_argument('--test_image_type', dest='test_image_type', default='.png', help='test image data type')

args = parser.parse_args()
#============================================================================================

def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    print 'Starting CycleGAN'
    with tf.Session(config=run_config) as sess:
        # model = cycleGAN(sess,
        #                     input_width_x=config.input_width_x, input_height_x=config.input_height_x,
        #                     input_width_y=config.input_width_y, input_height_y=config.input_height_y,
        #                     channel= config.channel_dim, lambda_val = config.lambda_val,
        #                     batch_size=config.batch_size, sample_size=config.sample_size,
        #                     learning_rate=config.learning_rate, momentum=config.momentum,
        #                     resnet_size=config.resnet_size,
        #                     logpoint= config.logpoint, checkpoint=config.checkpoint,
        #                     model=config.checkpoint_model, images=config.checkpoint_images
        #                     )
        model = cycleGAN(sess, args)

        if args.is_train:
            print 'Created CycleGAN Network...'
            print 'Start Training...'

            # model.train(config)
            model.train(args)
        else:
            print 'Generating Images...'
            # model.make_data(config)
            model.make_data(args)

main(args)

#Still Working....