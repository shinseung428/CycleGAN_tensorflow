import tensorflow as tf
import os
import numpy as np
import scipy.misc
from model_ import cycleGAN

configs = tf.app.flags

#Network Parameters
configs.DEFINE_integer("input_width_x", 128, "Input image width x [128]")
configs.DEFINE_integer("input_height_x", 128, "Input image height x [128]")
configs.DEFINE_integer("input_width_y", 128, "Input image width y [128]")
configs.DEFINE_integer("input_height_y", 128, "Input image width y [128]")
configs.DEFINE_integer("batch_size", 1, "Batch size [64]")
configs.DEFINE_integer("sample_size", 64, "Sample size [64]")
configs.DEFINE_integer("channel_dim", 1, "Image Channel [3]")
configs.DEFINE_string("resnet_size", "9_blocks", "Resnet size [9]")

#Training Parameters
configs.DEFINE_string("data_path", "./data", "Training datapath")
configs.DEFINE_string("trainsetA", "trainA", "Training Set A")
configs.DEFINE_string("trainsetB", "trainB", "Training Set B")
configs.DEFINE_string("image_type", "*.jpg", "jpeg image")

configs.DEFINE_integer("max_epochs", 25, "Max number of epochs [25]")
configs.DEFINE_integer("AB_G_iter", 1, "Generator_AB []")
configs.DEFINE_integer("BA_G_iter", 1, "Generator_BA []")
configs.DEFINE_integer("Dx_iter", 1, "Discriminator_x []")
configs.DEFINE_integer("Dy_iter", 1, "Discriminator_y []")
configs.DEFINE_integer("lambda_val", 10, "Lambda value [10]")

configs.DEFINE_float("learning_rate", 0.0002, "AdamOptim learning rate [0.0002]")
configs.DEFINE_float("momentum", 0.5, "Momentum [0.5]")

configs.DEFINE_integer("logpoint", 1000, "log point [100]")
configs.DEFINE_string("checkpoint", "./checkpoints", "Checkpoint folder [checkpoint]")
configs.DEFINE_string("checkpoint_model", "models", "Model folder [model]")
configs.DEFINE_string("checkpoint_images", "images", "Images folder [images]")

#Generate data using the model
configs.DEFINE_boolean("is_train", False, "Train model [True]")
configs.DEFINE_string("test_A", "trainA", "Synthetic images folder [synt_images]")
configs.DEFINE_string("test_path", "test_res", "Test result path [test_res]")
config = configs.FLAGS

def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    print 'Starting CycleGAN'
    with tf.Session(config=run_config) as sess:
        model = cycleGAN(sess,
                            input_width_x=config.input_width_x, input_height_x=config.input_height_x,
                            input_width_y=config.input_width_y, input_height_y=config.input_height_y,
                            channel= config.channel_dim, lambda_val = config.lambda_val,
                            batch_size=config.batch_size, sample_size=config.sample_size,
                            learning_rate=config.learning_rate, momentum=config.momentum,
                            resnet_size=config.resnet_size,
                            logpoint= config.logpoint, checkpoint=config.checkpoint,
                            model=config.checkpoint_model, images=config.checkpoint_images
                            )

        if config.is_train:
            print 'Created CycleGAN network'
            print 'Start Training'

            model.train(config)
        else:
            model.make_data(config)

main(config)

#Still Working....