import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Concatenate, Embedding, Dropout, LeakyReLU, Conv2DTranspose, Conv2D, Flatten,
                                     Reshape, Dense, Input, MaxPooling2D, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from numpy.random import randint, randn
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from osgeo import gdal


try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
except:
    pass


def load_image_data(path, img_dirs):
    """
    Load image data from given directory path and image subdirectories.
    :param path: Directory path where image folders are located.
    :param img_dirs: List of subdirectories containing images.
    :return: Tuple of image patches and corresponding labels.
    """
    images = []
    for img_dir in img_dirs:
        img_path = os.path.join(path, img_dir)
        images.append(img_path)

    ims = []
    labels = []
    for idx, img_dir in enumerate(images):
        files = glob.glob(os.path.join(img_dir, "*.tif"))
        print(f'Still working - class_{idx:04d}')
        label = np.asarray([idx] * len(files), dtype='uint8')
        labels.append(label)
        
        for file in files:
            dataset = gdal.Open(file)
            band1 = dataset.GetRasterBand(1)  # Red channel
            band2 = dataset.GetRasterBand(2)  # Green channel
            band3 = dataset.GetRasterBand(3)  # Blue channel
            img = np.dstack((band1.ReadAsArray(), band2.ReadAsArray(), band3.ReadAsArray()))
            ims.append(img)
    
    patches = np.array(ims)
    final_labels = np.concatenate(labels, axis=0)
    return patches, final_labels


def define_discriminator(input_shape=(224, 224, 3), n_classes=8):
    """
    Define the discriminator model.
    :param input_shape: Shape of input image.
    :param n_classes: Number of classes for the labels.
    :return: Compiled discriminator model.
    """
    in_label = Input(shape=(1,))
    label_embedding = Embedding(n_classes, 1500)(in_label)
    n_nodes = input_shape[0] * input_shape[1]  # 224x224
    label_embedding = Dense(n_nodes)(label_embedding)
    label_embedding = Reshape((input_shape[0], input_shape[1], 1))(label_embedding)

    in_image = Input(shape=input_shape)
    concat = Concatenate()([in_image, label_embedding])
    
    dis = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(concat)
    dis = LeakyReLU(alpha=0.2)(dis)
    dis = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(dis)
    dis = LeakyReLU(alpha=0.2)(dis)
    dis = Flatten()(dis)
    dis = Dropout(0.4)(dis)
    out_layer = Dense(1, activation='sigmoid')(dis)

    model = Model([in_image, in_label], out_layer)
    opt = Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model


def define_generator(latent_dim, n_classes=8):
    """
    Define the generator model.
    :param latent_dim: Latent space dimension.
    :param n_classes: Number of classes for the labels.
    :return: Generator model.
    """
    in_label = Input(shape=(1,))
    label_embedding = Embedding(n_classes, 224)(in_label)
    n_nodes = 7 * 7
    label_embedding = Dense(n_nodes)(label_embedding)
    label_embedding = Reshape((7, 7, 1))(label_embedding)

    in_latent = Input(shape=(latent_dim,))
    gen = Dense(128 * 7 * 7)(in_latent)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((7, 7, 128))(gen)

    concat = Concatenate()([gen, label_embedding])
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(concat)
    gen = LeakyReLU(alpha=0.2)(gen)
  
  	gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
  	gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same')(gen)
  	gen = LeakyReLU(alpha=0.2)(gen)

  	gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
  	gen = LeakyReLU(alpha=0.2)(gen)
  
  	gen = Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same')(gen)
	  gen = LeakyReLU(alpha=0.2)(gen)
  
    out_layer = Conv2D(3, (7, 7), activation='tanh', padding='same')(gen)

    model = Model([in_latent, in_label], out_layer)
    return model


def define_gan(generator, discriminator):
    """
    Define the GAN model combining generator and discriminator.
    :param generator: Generator model.
    :param discriminator: Discriminator model.
    :return: GAN model.
    """
    discriminator.trainable = False
    gen_noise, gen_label = generator.input
    gen_output = generator.output
    gen_output = discriminator([gen_output, gen_label])
    model = Model([gen_noise, gen_label], gen_output)
    opt = Adam(lr=0.0002, beta_1=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def load_real_samples(patches, labels):
    """
    Normalize real image data for training.
    :param patches: Array of image patches.
    :param labels: Array of corresponding labels.
    :return: Normalized images and labels.
    """
    X = patches.astype('float32')
    X = (X - 127.5) / 127.5
    return X, labels


def generate_real_samples(dataset, n_samples):
    """
    Generate a batch of real samples for training.
    :param dataset: Tuple of images and labels.
    :param n_samples: Number of samples to generate.
    :return: Real image samples and labels.
    """
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = np.ones((n_samples, 1))  # Label=1 for real images
    return [X, labels], y


def generate_latent_points(latent_dim, n_samples, n_classes=8):
    """
    Generate latent points for the generator.
    :param latent_dim: Dimension of the latent space.
    :param n_samples: Number of samples to generate.
    :param n_classes: Number of classes for labels.
    :return: Latent space samples and random labels.
    """
    z_input = randn(latent_dim * n_samples).reshape(n_samples, latent_dim)
    labels = randint(0, n_classes, n_samples)
    return z_input, labels


def generate_fake_samples(generator, latent_dim, n_samples):
    """
    Generate fake samples from the generator.
    :param generator: The generator model.
    :param latent_dim: Dimension of the latent space.
    :param n_samples: Number of fake samples to generate.
    :return: Fake image samples and corresponding labels.
    """
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    images = generator.predict([z_input, labels_input])
    y = np.zeros((n_samples, 1))  # Label=0 for fake images
    return images, labels_input, y


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    """
    Train the GAN model.
    :param g_model: Generator model.
    :param d_model: Discriminator model.
    :param gan_model: GAN model combining generator and discriminator.
    :param dataset: Training dataset (images and labels).
    :param latent_dim: Latent space dimension.
    :param n_epochs: Number of epochs to train.
    :param n_batch: Batch size for training.
    """
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    
    disrealloss, disfakeloss, generatorloss = [], [], []
    
    for epoch in range(n_epochs):
        for batch in range(bat_per_epo):
            # Train discriminator on real and fake images
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            d_loss_real = d_model.train_on_batch([X_real, labels_real], y_real)
            disrealloss.append(d_loss_real)

            [X_fake, labels_fake, y_fake] = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss_fake = d_model.train_on_batch([X_fake, labels_fake], y_fake)
            disfakeloss.append(d_loss_fake)

            # Train the generator via the discriminator's error
            z_input, labels_input = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))  # Labels for fake samples
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            generatorloss.append(g_loss)
            
            # Print batch training status
            print(f'Epoch {epoch+1}/{n_epochs}, Batch {batch+1}/{bat_per_epo}, '
                  f'Dis Real Loss: {d_loss_real:.9f}, Dis Fake Loss: {d_loss_fake:.9f}, Gen Loss: {g_loss:.9f}')

    # Save the trained generator model
    g_model.save("generator_model.h5")
    return disrealloss, disfakeloss, generatorloss


def save_generated_images(generator, latent_dim, n_samples, output_dir):
    """
    Generate and save images produced by the trained generator.
    :param generator: Trained generator model.
    :param latent_dim: Latent space dimension.
    :param n_samples: Number of images to generate.
    :param output_dir: Directory to save the generated images.
    """
    latent_points, labels = generate_latent_points(latent_dim, n_samples)
    labels = np.array([5] * n_samples)  # Specify label for the images

    generated_images = generator.predict([latent_points, labels])
    generated_images = (generated_images + 1) / 2.0  # Rescale to [0, 1]
    generated_images = (generated_images * 255).astype(np.uint8)
    
    for i in range(generated_images.shape[0]):
        img = generated_images[i]
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(output_dir, f'generated_image_{i:04d}.tif'))


# --- Main execution ---

if __name__ == "__main__":
    path = "/home/alexandra/home/alexandra/"
    img_dirs = ['ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
    
    patches, labels = load_image_data(path, img_dirs)
    dataset = load_real_samples(patches, labels)
    
    latent_dim = 1500
    d_model = define_discriminator()
    g_model = define_generator(latent_dim)
    gan_model = define_gan(g_model, d_model)
    
    disrealloss, disfakeloss, generatorloss = train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=300)
    
    # Save losses to disk
    with open("disreal_loss.dat", "wb") as f:
        pickle.dump(disrealloss, f)
    with open("disfake_loss.dat", "wb") as f:
        pickle.dump(disfakeloss, f)
    with open("generator_loss.dat", "wb") as f:
        pickle.dump(generatorloss, f)
    
    # Save generated images
    save_generated_images(g_model, latent_dim, 1500, "/home/alexandra/home/alexandra/generated_images/")
