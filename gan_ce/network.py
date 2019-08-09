import os
import cv2
import numpy as np
from copy import deepcopy
import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer
import gan_ce.network_utils as nu

class Network:
    def __init__(self, tiles=(2,2), shape=(256,256,3), epsilon=1e-10):
        # Set tile number and size
        self.shape = shape
        self.tiles = tiles
        # Calculate the size of the input image
        self.image_resize_to = (tiles[0]*shape[0],tiles[1]*shape[1])

        # Reset old session stuff because of the recovery bug (see https://github.com/tflearn/tflearn/issues/527)
        tf.reset_default_graph()
        
        # Create a flag to set is is training phase or not
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        # Discriminator input for ground truth
        self.x_in = tf.placeholder(tf.float32, [None, shape[0], shape[1], shape[2]])
        # Generator input for masked images 
        self.masked_x_in = tf.placeholder(tf.float32, [None, shape[0], shape[1], shape[2]])

        # Creat the Generator
        self.generator = self.create_generator(self.masked_x_in)
        # Create the adversial discriminator with input for ground truth
        self.real_discriminator = self.create_discriminator(self.x_in, reuse=tf.AUTO_REUSE)
        # Reuse the created adversial discriminator and set the Generator output as Input (for the fake images)
        self.fake_discriminator = self.create_discriminator(self.generator, reuse=tf.AUTO_REUSE)

        # Define the discriminator loss by calculating the adverssarial los (like in the paper)
        self.discriminator_loss = -tf.reduce_mean(tf.log(self.real_discriminator + epsilon) + tf.log(1 - self.fake_discriminator + epsilon))
        # Define the generator loss by calculating the joint loss of GAN loss and L2 reconstruction loss
        self.generator_loss = -tf.reduce_mean(tf.log(self.fake_discriminator + epsilon)) + 100*tf.reduce_mean(tf.reduce_sum(tf.square(self.x_in - self.generator), [1, 2, 3]))

        # Define a Adam Optimizer to train the disciminator with the respective loss
        self.discriminator_optimizer = tf.train.AdamOptimizer(2e-4).minimize(self.discriminator_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "discriminator"))
        # Define a Adam Optimizer to train the generator with the respective loss
        self.generator_optimizer = tf.train.AdamOptimizer(2e-4).minimize(self.generator_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator"))

        # Create tensorflow session and 
        self.sess = tf.Session()
        # Initialize instanced variables
        self.sess.run(tf.global_variables_initializer())
		
    def load_weights(self, weights_path='./weights/weights.ckpt'):
        if not os.path.isdir(os.path.dirname(weights_path)):
            raise Exception("Cannot finde weights on path '" + weights_path + "'")
        # Load the weights for the generator and discriminator
        saver = tf.train.Saver()
        saver.restore(self.sess, weights_path)
        print('Weights loaded.')
		
    def load_weights_generator(self, weights_path='./weights/weights.ckpt'):
        if not os.path.isdir(os.path.dirname(weights_path)):
            raise Exception("Cannot finde weights on path '" + weights_path + "'")
        # Load the weights for the generator only
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator"))
        saver.restore(self.sess, weights_path)

    def create_generator(self, input):
        # Create a generator model as described in the paper
        with tf.variable_scope("generator", reuse=None):
            ## Encoder
            x = nu._leaky_relu(nu._conv2d(input, "conv1", filters=64, kernel_size=4,  strides=2, padding="same"))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d(x, "conv2", filters=64, kernel_size=4, strides=2, padding="same"), "conv2", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d(x, "conv3", filters=128, kernel_size=4, strides=2, padding="same"), "conv3", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d(x, "conv4", filters=256, kernel_size=4, strides=2, padding="same"), "conv4", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d(x, "conv5", filters=512, kernel_size=4, strides=2, padding="same"), "conv5", is_training=self.is_training))
            ## Bottleneck
            x = nu._fully_connected_2d(x, "fc1")
            ## Decoder
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d_transpose(x, "conv_trans1", filters=512, kernel_size=4, strides=2, padding="same"), "conv_trans1", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d_transpose(x, "conv_trans2", filters=256, kernel_size=4, strides=2, padding="same"), "conv_trans2", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d_transpose(x, "conv_trans3", filters=128, kernel_size=4, strides=2, padding="same"), "conv_trans3", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d_transpose(x, "conv_trans4", filters=64, kernel_size=4, strides=2, padding="same"), "conv_trans4", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d_transpose(x, "conv_trans5", filters=64, kernel_size=4, strides=2, padding="same"), "conv_trans5", is_training=self.is_training))
            ## Output
            x = tf.nn.tanh(nu._leaky_relu(nu._conv2d(x, "conv_out", filters=self.shape[2], kernel_size=4, strides=1, padding="same")))
            return x

    def create_discriminator(self, input,  reuse=None):
        # Create a discriminator model as described in the paper
        with tf.variable_scope("discriminator", reuse=reuse):
            x = nu._leaky_relu(nu._conv2d(input, "conv1", filters=64, kernel_size=4,  strides=2, padding="same"))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d(x, "conv2", filters=64, kernel_size=4, strides=2, padding="same"), "conv2", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d(x, "conv3", filters=128, kernel_size=4, strides=2, padding="same"), "conv3", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d(x, "conv4", filters=256, kernel_size=4, strides=2, padding="same"), "conv4", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d(x, "conv5", filters=512, kernel_size=4, strides=2, padding="same"), "conv5", is_training=self.is_training))
            x = nu._fully_connected(x, "fc2")
            x = tf.sigmoid(x)
            return x

    def train(self, images=[], epochs=50000, batch_size=4, weights_path='./weights/weights.ckpt.index', saving_epochs=100, mask_min_rectangles=1, mask_max_rectangles=3, mask_min_lines=1, mask_max_lines=3, mask_min_circles=1, mask_max_circles=3):
        # Create tiles out of the training images and normalize to -1 to 1
        saver = tf.train.Saver()
        image_tiles = []
        for image in images:
            image = cv2.resize(image, (self.image_resize_to[1], self.image_resize_to[0]))
            for r in range(0, image.shape[0], self.shape[0]):
                for c in range(0, image.shape[1], self.shape[1]):
                    image_tiles.append(image[r:r+self.shape[0], c:c+self.shape[1]] / 127.5 - 1.0)

        # Remove unneded image to save memory          
        images = None

        # Training loop
        for epoch in range(0, epochs):
            # Create two batches for masked images (input for Generator) and a batch with the ground truth (for the "real Disciminator")
            masked_batch = np.zeros([batch_size, self.shape[0], self.shape[1], self.shape[2]])
            ground_truth_batch = np.zeros([batch_size, self.shape[0], self.shape[1], self.shape[2]])

            # Now we fill the batches with n random tiles
            batch_idx = 0
            for index in np.random.randint(0, len(image_tiles), [batch_size]):
                # Create a random mask
                random_mask = np.zeros([self.shape[0], self.shape[1]])
                # Draw rectangles
                if mask_max_rectangles>0 and mask_max_rectangles<mask_min_rectangles:
                    for _ in range(np.random.randint(mask_min_rectangles, mask_max_rectangles)):
                        X1, Y1 = np.random.randint(0, self.shape[0]), np.random.randint(0, self.shape[1])
                        X2, Y2 = np.random.randint(0, self.shape[0]), np.random.randint(0, self.shape[1])
                        cv2.rectangle(random_mask, (X1, Y1), (X2, Y2), (1, 1, 1), cv2.FILLED)
                # Drawlines with a width between 3 and 10% of the height
                if mask_max_lines>0 and mask_max_lines<mask_min_lines:
                    for _ in range(np.random.randint(mask_min_lines, mask_max_lines)):
                        X1, Y1 = np.random.randint(0, self.shape[0]), np.random.randint(0, self.shape[1])
                        radius = np.random.randint(3, int(self.shape[0] * 0.1))
                        cv2.circle(random_mask, (X1, Y1), radius, (1, 1, 1), -1)
                # Draw circles with a width between 3 and 20% of the height
                if mask_max_circles>0 and mask_max_circles<mask_min_circles:
                    for _ in range(np.random.randint(mask_min_circles, mask_max_circles)):
                        X1, Y1 = np.random.randint(0, self.shape[0]), np.random.randint(0, self.shape[1])
                        radius = np.random.randint(3, int(self.shape[0] * 0.2))
                        cv2.circle(random_mask, (X1, Y1), radius, (1, 1, 1), -1)
                random_mask = np.dstack((random_mask, random_mask, random_mask))

                # Add the tile into the batch and mask the areas that should be inpainted by setting them to [1,1,1]
                masked_batch[batch_idx] = deepcopy(image_tiles[index])
                masked_batch[batch_idx][np.where((random_mask==[1, 1, 1]).all(axis=2))] = [1, 1, 1]
                # Add the same tile as ground truth for the Disciminator
                ground_truth_batch[batch_idx] = deepcopy(image_tiles[index])
                batch_idx += 1

            # Train the Disciminator
            self.sess.run(self.discriminator_optimizer, feed_dict={self.masked_x_in: masked_batch, self.x_in: ground_truth_batch, self.is_training:True})
            # Train the Generator
            self.sess.run(self.generator_optimizer, feed_dict={self.masked_x_in: masked_batch, self.x_in: ground_truth_batch, self.is_training:True})
            # Get the Discriminator and Generator loss 
            [discriminator_loss, generator_loss, predicted_image] = self.sess.run([self.discriminator_loss, self.generator_loss, self.generator],
                        feed_dict={self.masked_x_in: masked_batch, self.x_in: ground_truth_batch, self.is_training: False})

            # Print the current epoch and losses
            print(("epoch: %d, Discriminator_loss: %g, Generator_loss: %g" % (epoch, discriminator_loss, generator_loss)))
            
            # Save the weights if the specified epoch reached
            if (epoch + 1) % saving_epochs == 0:
                print("Save weights")
                saver.save(self.sess, weights_path)

    def predict(self, image, mask):
        # Create a copy of the unmodified image and mask
        ori_image = deepcopy(image)
        ori_mask = deepcopy(mask)
        # Remember that shape 
        ori_height, ori_width, ori_channels = image.shape
        # Resize the image and mask to the needed shape
        image = cv2.resize(image, (self.image_resize_to[1], self.image_resize_to[0]))
        mask = cv2.resize(mask, (self.image_resize_to[1], self.image_resize_to[0]))

        # Do the prediction 
        result = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype = "uint8")
        for r in range(0, image.shape[0], self.shape[0]):
            for c in range(0,image.shape[1], self.shape[1]):
                # Extract a tile out of the image and normalize to -1 to 1
                chunked_image = image[r:r+self.shape[0], c:c+self.shape[1]] / 127.5 - 1.0
                chunked_mask = mask[r:r+self.shape[0], c:c+self.shape[1]]
                # Mask the image by setting the inpainting regions to [1,1,1]
                chunked_image[np.where((chunked_mask==[1, 1, 1]).all(axis=2))] = [1, 1, 1]
                # Create a batch with the masked image 
                input_image_masked = np.zeros([1, self.shape[0], self.shape[1], self.shape[2]])
                input_image_masked[0, :, :, :] = chunked_image
                # Let the Generator create a prediction
                predicted_image = self.sess.run(self.generator, feed_dict={self.masked_x_in: input_image_masked, self.is_training: False})
                # Put the predicted tile onto the result image
                result[r:r+self.shape[0], c:c+self.shape[1]] = ((predicted_image[0, :, :, :] + 1) * 127.5).astype(np.uint8)

        # Resize the result image to the input shape
        result = cv2.resize(result, (ori_width, ori_height))
        # Get the regions where inpainting was declared by the mask
        idx = (ori_mask == 1)
        # Overwrite the damaged regions that have to be inpainted 
        ori_image[idx] = result[idx]
        # Return the result
        return ori_image
