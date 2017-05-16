import tensorflow as tf
import numpy as np
import os
from PIL import Image
from glob import glob

from architecture import *
from utils import *

class cycleGAN():
    # def __init__(self,
    #              sess,
    #              input_width_x, input_height_x,
    #              input_width_y, input_height_y,
    #              channel, lambda_val,
    #              batch_size, sample_size,
    #              learning_rate, momentum,
    #              resnet_size,
    #              logpoint, checkpoint,
    #              model, images):
    #     self.sess = sess

    #     self.input_width_x = input_width_x
    #     self.input_height_x = input_height_x
    #     self.input_width_y = input_width_y
    #     self.input_height_y = input_height_y
    #     self.channel = channel
    #     self.df_dim = 64
    #     self.batch_size = batch_size
    #     self.sample_size = sample_size

    #     self.lambda_val = lambda_val
    #     self.learning_rate = learning_rate
    #     self.momentum = momentum

    #     self.resnet_size = resnet_size

    #     self.logpoint = logpoint
    #     self.model_path = checkpoint + "/" + model + "/"

    #     self.images_path = checkpoint + "/" + images + "/"

    #     if self.channel == 1:
    #         self.is_gray=True
    #     else:
    #         self.is_gray=False

    #     self.input_dim_x = [self.input_height_x, self.input_width_x, self.channel]
    #     self.input_dim_y = [self.input_height_y, self.input_width_y, self.channel]

    #     self.debug = 0

    #     self.build_model()
    #     self.build_losses()
    def __init__(self,sess,args):
        self.sess = sess

        self.input_width_x = args.input_width_x
        self.input_height_x = args.input_height_x
        self.input_width_y = args.input_width_y
        self.input_height_y = args.input_height_y
        self.channel = args.channel_dim
        self.df_dim = 64
        self.batch_size = args.batch_size
        self.sample_size = args.sample_size

        self.lambda_val = args.lambda_val
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum

        self.resnet_size = args.resnet_size

        self.logpoint = args.logpoint
        self.model_path = args.log_folder + "/" + args.log_models + "/"

        self.images_path = args.log_folder + "/" + args.log_images + "/"

        if self.channel == 1:
            self.is_gray=True
        else:
            self.is_gray=False

        self.input_dim_x = [self.input_height_x, self.input_width_x, self.channel]
        self.input_dim_y = [self.input_height_y, self.input_width_y, self.channel]

        self.debug = 0

        self.build_model()
        self.build_losses()
    def build_model(self):

        #make placholders
        self.x_input = tf.placeholder(tf.float32, [self.batch_size] + self.input_dim_x, name = 'x_input')
        self.y_input = tf.placeholder(tf.float32, [self.batch_size] + self.input_dim_y, name = 'y_input')
        self.x_samples = tf.placeholder(tf.float32, [self.sample_size] + self.input_dim_x, name = 'x_samples')
        self.y_samples = tf.placeholder(tf.float32, [self.sample_size] + self.input_dim_y, name = 'y_samples')

        images_x = self.x_input
        images_y = self.y_input
        sample_x = self.x_samples
        sample_y = self.y_samples

        #make network models
        print 'build generator_xy'
        self.G_xy = self.generator(images_x, name="generator_xy")
        print 'build discriminator_y'
        self.D_y, self.D_y_logits = self.discriminator(images_y, name="discriminator_y")
        self.D_y_, self.D_y_logits_ = self.discriminator(self.G_xy, name="discriminator_y", reuse=True)

        print 'build generator_yx'
        self.G_yx = self.generator(images_y, name="generator_yx")
        print 'build discriminator_x'
        self.D_x, self.D_x_logits = self.discriminator(images_x, name="discriminator_x")
        self.D_x_, self.D_x_logits_ = self.discriminator(self.G_yx, name="discriminator_x", reuse=True)

        print 'build generator_xyx'
        self.G_xyx = self.generator(self.G_xy, name="generator_yx", reuse=True)
        print 'build generator_yxy'
        self.G_yxy = self.generator(self.G_yx, name="generator_xy", reuse=True)

        self.d_sum_y = tf.summary.histogram("d_y", self.D_y)
        self.d_sum_y_ = tf.summary.histogram("d_y_", self.D_y_)
        #self.images_G = tf.summary.merge([tf.summary.image("x", images_x),
        #                                  tf.summary.image("G_xy", self.G_xy),
        #                                  tf.summary.image("G_xyx", self.G_xyx)])

        self.images_x = tf.concat(0,[images_x, self.G_xy, self.G_xyx])
	self.images_x_sum =tf.summary.image("x", self.images_x, max_outputs=3)
        #self.orig_img_x = tf.summary.image("x", images_x, max_outputs=3)
        #self.G_sum_xy = tf.summary.image("x", self.G_xy, max_outputs=3)
        #self.G_sum_xyx = tf.summary.image("x", self.G_xyx, max_outputs=3)

        self.d_sum_x = tf.summary.histogram("d_x", self.D_x)
        self.d_sum_x_ = tf.summary.histogram("d_x_", self.D_x_)
        #self.images_D = tf.summary.merge([tf.summary.image("y", images_y),
        #                                  tf.summary.image("G_yx", self.G_yx),
        #                                  tf.summary.image("G_yxy", self.G_yxy)])

	self.images_y = tf.concat(0,[images_y, self.G_yx, self.G_yxy])
	self.images_y_sum = tf.summary.image("y", self.images_y, max_outputs=3)
        #self.orig_img_y = tf.summary.image("y", images_y, max_outputs=3)
        #self.G_sum_yx = tf.summary.image("y", self.G_yx, max_outputs=3)
        #self.G_sum_yxy = tf.summary.image("y", self.G_yxy, max_outputs=3)
        # self.G_sum_yx = tf.summary.image("G_yx", self.G_yx)

        if self.debug:
            self.g_loss_y = tf.Print(self.g_loss_y, [self.G_xy], "G_xy : ", summarize=10)
            # self.g_loss_y = tf.Print(self.g_loss_y, [tf.nn.sparse_softmax_cross_entropy_with_logits(
            #                                 logits=self.D_y_logits_, labels=ones_label(self.D_y_)
            #                               )], "sparse_softmax_cross_entropy_with_logits\n")

            # self.d_loss_y = tf.Print(self.d_loss_y, [self.d_loss_real_y], message="\nmse_real : \n")
            # self.d_loss_y = tf.Print(self.d_loss_y, [self.d_loss_fake_y], message="\nmse_fake : \n")
            # self.d_loss_y = tf.Print(self.d_loss_y, [self.d_loss_y], message="\nmse_loss : \n")
            # self.d_loss_y = tf.Print(self.d_loss_y, [self.D_y_logits_], message="\n\nD_y_logits_ : \n")

        #gets all the variables initialized with trainable set to True
        trainable_vars = tf.trainable_variables()

        self.D_vars_x = [vars for vars in trainable_vars if 'discriminator_x' in vars.name]
        self.D_vars_y = [vars for vars in trainable_vars if 'discriminator_y' in vars.name]
        self.G_vars_all = [vars for vars in trainable_vars if 'generator' in vars.name]

        for v in self.G_vars_all:
            print v
        print ' '
        for w in self.D_vars_x:
            print w
        print ' '
        for w in self.D_vars_y:
            print w
        # input("pause")
    def build_losses(self):
            def ones(layer):
                return tf.ones_like(layer, dtype=tf.float32)

            def zeros(layer):
                return tf.zeros_like(layer, dtype=tf.float32)

            def get_loss(input, label):
                w = int(input.get_shape()[1])
                h = int(input.get_shape()[2])
                if label == 'ones':
                    return tf.reduce_mean(tf.pow(input - ones(input), 2 * ones(input)), [1, 2, 3])
                else:
                    return tf.reduce_mean(tf.pow(input - zeros(input), 2 * ones(input)), [1, 2, 3])

            self.d_loss_real_x = get_loss(self.D_x, label='ones')
            self.d_loss_fake_x = get_loss(self.D_x_, label='zeros')
            self.d_loss_x = self.d_loss_fake_x + self.d_loss_real_x

            self.d_loss_real_y = get_loss(self.D_y, label='ones')
            self.d_loss_fake_y = get_loss(self.D_y_, label='zeros')
            self.d_loss_y = self.d_loss_real_y + self.d_loss_fake_y

            # GENERATOR G_xy LOSS(MSE)
            self.g_loss_y = get_loss(self.D_y_, label='ones')

            # GENERATOR G_yx LOSS(MSE)
            self.g_loss_x = get_loss(self.D_x_, label='ones')

            # Cyclic Losses(L1 loss)
            self.g_loss_xyx = tf.reduce_mean(tf.abs(self.G_xyx - self.x_input), [1, 2, 3])
            self.g_loss_yxy = tf.reduce_mean(tf.abs(self.G_yxy - self.y_input), [1, 2, 3])

            self.g_loss = self.g_loss_x + self.g_loss_y * \
                                          self.lambda_val * (self.g_loss_xyx + self.g_loss_yxy)

            # self.d_loss_real_y_sum = tf.summary.scalar("d_loss_real_y", tf.reduce_mean(self.d_loss_real_y))
            # self.d_loss_fake_y_sum = tf.summary.scalar("d_loss_fake_y", tf.reduce_mean(self.d_loss_fake_y))
            self.d_loss_y_sum = tf.summary.scalar("d_loss_y", tf.reduce_mean(self.d_loss_y))
            self.g_loss_y_sum = tf.summary.scalar("g_loss_y", tf.reduce_mean(self.g_loss_y))
            self.d_loss_x_sum = tf.summary.scalar("d_loss_x", tf.reduce_mean(self.d_loss_x))
            self.g_loss_x_sum = tf.summary.scalar("g_loss_x", tf.reduce_mean(self.g_loss_x))
            self.g_loss_sum = tf.summary.scalar("g_loss", tf.reduce_mean(self.g_loss))


    #training starts here
    def train(self,args):
        saver = tf.train.Saver()
        #make optimizers
        D_optimizer_x = tf.train.AdamOptimizer(self.learning_rate, beta1=self.momentum, name='Adam_Dx').minimize(self.d_loss_x, var_list=self.D_vars_x)
        D_optimizer_y = tf.train.AdamOptimizer(self.learning_rate, beta1=self.momentum, name='Adam_Dy').minimize(self.d_loss_y, var_list=self.D_vars_y)
        # D_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.momentum, name='Adam_Dx').minimize(self.d_loss_y, var_list=self.D_vars_all)

        # G_optimizer_xy = tf.train.AdamOptimizer(self.learning_rate, beta1=self.momentum, name='Adam_Gx').minimize(self.g_loss_x, var_list=self.Gx_vars)
        # G_optimizer_yx = tf.train.AdamOptimizer(self.learning_rate, beta1=self.momentum, name='Adam_Gy').minimize(self.g_loss_y, var_list=self.Gy_vars)
        G_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.momentum, name='Adam_Gx').minimize(self.g_loss, var_list=self.G_vars_all)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        image_paths_A = glob(os.path.join(args.data_path, args.trainsetA, args.image_type))
        image_paths_B = glob(os.path.join(args.data_path, args.trainsetB, args.image_type))
        # joint_file = read_joints(os.path.join(args.data_path, args.joint_path))

        #prepare sample images
        sample_filesA = image_paths_A[0:self.sample_size]
        sample_filesB = image_paths_B[0:self.sample_size]
        sample_imagesA = [load_image(file_name,
                                     input_width=self.input_width_x,
                                     input_height=self.input_height_x,
                                     is_gray=self.is_gray
                                     ) for file_name in sample_filesA]
        sample_imagesB = [load_image(file_name,
                                     input_width=self.input_width_y,
                                     input_height=self.input_height_y,
                                     is_gray=self.is_gray
                                     ) for file_name in sample_filesB]

        #variables to show in Summary
        self.g_sum = tf.summary.merge([self.images_x_sum,
				       self.images_y_sum,
                                       self.g_loss_y_sum,
                                       self.g_loss_x_sum,
                                       self.g_loss_sum])
        self.d_sum_y = tf.summary.merge([self.d_sum_y,
                                       self.d_sum_y_,
                                       self.d_loss_y_sum])
        self.d_sum_x = tf.summary.merge([self.d_sum_x,
                                       self.d_sum_x_,
                                       self.d_loss_x_sum])

        writer = tf.summary.FileWriter('./graphs', self.sess.graph)
        counter = 0

        #Training starts
        for epoch in xrange(args.max_epochs):
            batch_indices = min(len(image_paths_A), np.inf) // self.batch_size

            for index in xrange(0, batch_indices):
                batch_imagesA = image_paths_A[index*self.batch_size:(index+1)*self.batch_size]
                batch_imagesB = image_paths_B[index*self.batch_size:(index+1)*self.batch_size]

                batchA = [load_image(file_name,
                                     input_width=self.input_width_x,
                                     input_height=self.input_height_x,
                                     resize_width=self.input_width_x,
                                     resize_height=self.input_height_x,
                                     is_gray=self.is_gray) for file_name in batch_imagesA]
                batchB = [load_image(file_name,
                                     input_width=self.input_width_y,
                                     input_height=self.input_height_y,
                                     resize_width=self.input_width_y,
                                     resize_height=self.input_height_y,
                                     is_gray=self.is_gray) for file_name in batch_imagesB]

                if self.is_gray:
                    batchA = np.expand_dims(np.array(batchA).astype(np.float32), axis=3)
                    batchB = np.expand_dims(np.array(batchB).astype(np.float32), axis=3)
                else:
                    batchA = np.array(batchA).astype(np.float32)
                    batchB =np.array(batchB).astype(np.float32)

                #Train generator
                _,summary = self.sess.run([G_optimizer,self.g_sum], feed_dict = {self.x_input: batchA, self.y_input: batchB})
                writer.add_summary(summary, counter)

                #Train discriminator x
                _,summary = self.sess.run([D_optimizer_x,self.d_sum_x], feed_dict = {self.x_input: batchA, self.y_input: batchB})
                writer.add_summary(summary, counter)

                #Train discriminator y
                _, summary = self.sess.run([D_optimizer_y, self.d_sum_y], feed_dict={self.x_input: batchA, self.y_input: batchB})
                writer.add_summary(summary, counter)

                d_x_err = self.d_loss_x.eval({self.x_input: batchA, self.y_input: batchB})
                d_y_err = self.d_loss_y.eval({self.x_input: batchA, self.y_input: batchB})
                generator_err = self.g_loss.eval({self.x_input: batchA, self.y_input: batchB})

                print ("Epoch [%2d] step [%2d]\n Generator_error: [%.8f] \n Discriminator_x_error: [%.8f]\n Discriminator_y_error: [%.8f]\n" %
                       (epoch, index, generator_err, d_x_err, d_y_err))


                if counter%self.logpoint == 0:
                    saver.save(self.sess,self.model_path + 'model_{:06}'.format(counter))
                    generated_img = self.G_xy.eval({self.x_input:batchA})
                    cycled_img = self.G_xyx.eval({self.x_input:batchA})
                    save_images(batchA, generated_img, cycled_img, self.images_path + 'model_{:06}'.format(counter) + ".jpg")
                    print ' Saved model'

                counter += 1
                input("pause")

    def generator(self, images, reuse=False, name="generator"):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            layers = []
            # convolution1
            conv1 = tf.nn.relu(instance_norm(conv2d_(images, self.channel, 64,
                                                    kernel=7, strides=1, padding=3,
                                                    name = 'conv1')))
            layers.append(conv1)

            # convolution2
            conv2 = tf.nn.relu(instance_norm(conv2d_(conv1, 64, 128,
                                                    kernel=3, strides=2, padding=1,
                                                    name = 'conv2')))
            layers.append(conv2)

            # convolution3
            conv3 = tf.nn.relu(instance_norm(conv2d_(conv2, 128, 256,
                                                    kernel=3, strides=2, padding=1,
                                                    name = 'conv3')))
            layers.append(conv3)

            n_layers = 9 if self.resnet_size == "9_blocks" else 6
            for i in range(1,n_layers+1):
                resnet_ = resnet(layers[-1], 256, kernel=3, strides=1, name='resnet_%d'%i,reuse=reuse)
                layers.append(resnet_)

            deconv1 = tf.nn.relu(instance_norm(deconv2d(layers[-1], 256, 128,
                                                             kernel=3, strides=2,
                                                             name='deconv1')))
            layers.append(deconv1)

            # deconv2
            deconv2 = tf.nn.relu(instance_norm(deconv2d(deconv1, 128, 64,
                                                             kernel=3, strides=2,
                                                             name='deconv2')))
            layers.append(deconv2)

            deconv3 = tf.nn.tanh(deconv2d(deconv2, 64, self.channel,
                                         kernel=7, strides=1,
                                         name='deconv3'))
            layers.append(deconv3)

            return deconv3

    def discriminator(self, images, reuse=False, name ="discriminator"):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            n_layers = 3
            layers = []

            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            # input = tf.concat([images, self.input_dim_y], 3)
            kw = 4
            padw = int(np.ceil((kw-1)/2))
            # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
            # with tf.variable_scope("layer_1"):
            convolved = conv2d_(images, self.channel, self.df_dim,
                               kernel=kw, strides=2, padding=padw,
                               name="conv1")
            rectified = leaky_relu(convolved, 0.2)
            layers.append(rectified)


            nf_mult=1
            nf_mult_prev=1
            for i in range(1,n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2**i,8)
                convolved = conv2d_(layers[-1], self.df_dim * nf_mult_prev, self.df_dim*nf_mult,
                                    kernel=kw, strides=2, padding=padw,
                                    name="conv%d"%(len(layers)+1))
                normalized = batchnorm(convolved,name="batchnorm%d"%(len(layers)+1))
                rectified = leaky_relu(normalized, 0.2)
                layers.append(rectified)

            convolved = conv2d_(rectified, self.df_dim * nf_mult, self.channel,
                                kernel=kw, strides=1, padding=padw,
                                name="conv%d"%(len(layers)+1))
            output = tf.sigmoid(convolved)
            layers.append(output)

            return output, convolved


    def make_data(self,args):
        checkpoint_path = args.log_folder + "/" + args.log_models + "/"
        meta_paths = sorted(glob(os.path.join(checkpoint_path, "*.meta")),key=os.path.getmtime)
        
        # new_saver = tf.train.import_meta_graph(meta_paths[len(meta_paths)-1])
        new_saver = tf.train.Saver()
        new_saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_path))
        image_pathsA = sorted(glob(os.path.join(args.data_path, args.test_A, args.image_type)),key=os.path.getmtime)
        image_pathsB = sorted(glob(os.path.join(args.data_path, args.test_B, args.image_type)),key=os.path.getmtime)
        print 'loaded model ', tf.train.latest_checkpoint(checkpoint_path)

        batchA = [load_image(file_name,
                             input_width=self.input_width_x,
                             input_height=self.input_height_x,
                             resize_width=self.input_width_x,
                             resize_height=self.input_height_x,
                             is_gray=self.is_gray) for file_name in image_pathsA]
        batchB = [load_image(file_name,
                             input_width=self.input_width_x,
                             input_height=self.input_height_x,
                             resize_width=self.input_width_x,
                             resize_height=self.input_height_x,
                             is_gray=self.is_gray) for file_name in image_pathsB]

        # jointfile = "./hand_generator/joint_partial.txt"
        # joints_ = open(jointfile).read().split('\n')

        for index,image in enumerate(batchA):
            print "generating image # %d"%index
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=3)
            generated_img = self.G_xy.eval({self.x_input: image})

            imageA = np.expand_dims(batchA[index], axis=0)
            imageA = np.expand_dims(imageA, axis=3)
            imageB = np.expand_dims(batchB[index], axis=0)
            imageB = np.expand_dims(imageB, axis=3)

            A_test_path = os.path.join(args.data_path, args.A_path)
            B_test_path = os.path.join(args.data_path, args.B_path)
            fake_test_path = os.path.join(args.data_path, args.test_path)

            save_image(imageA, A_test_path + "/synt_{:06}".format(index)+args.test_image_type)
            save_image(imageB, B_test_path + "/real_{:06}".format(index)+args.test_image_type)
            save_image(generated_img, fake_test_path + "/fake_{:06}".format(index)+args.test_image_type)
            input("pause")