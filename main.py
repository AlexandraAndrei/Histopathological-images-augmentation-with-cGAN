from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from numpy.random import randint
from numpy.random import randn
from numpy import ones
from numpy import zeros
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from numpy import asarray
from numpy.random import randint
from keras.models import load_model
import numpy as np
from numpy import asarray

try:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
	pass
# Read data
path = "/home/alexandra/home/alexandra/"
img_dir=['ADI', 'DEB', 'LYM', 'MUC','MUS','NORM','STR', 'TUM']

images = []
for i in range(0, len(img_dir)):
	input_ = os.path.join(path, img_dir[i])
	images.append(input_)

files = []
ims = []
img = []
label = []

for a in range(0,len(images)):
	files = glob.glob(images[a]+"/*.tif")
	globals()[f"nummber_of_records{a}"] = len(files)
	globals()[f'label{a}'] = np.asarray([a]*len(files), dtype='uint8')
    # label_{"%d"}, a=len(files)
    # 'label_' + str(a)= len(files)
	print('Still working - class_{:04d}'.format(a))
	for j in range(0, len(files)):
		dataset = gdal.Open(files[j])
        # since there are 3 bands
# we store in 3 different variables
		band1 = dataset.GetRasterBand(1)  # Red channe   	
		band2 = dataset.GetRasterBand(2)  # Green channel
		band3 = dataset.GetRasterBand(3)  # Blue channel
		b1 = band1.ReadAsArray()
		b2 = band2.ReadAsArray()
		b3 = band3.ReadAsArray()
		img = np.dstack((b1, b2, b3))
		ims.append(img)

# vector containing all images
patches = np.array(ims)

# labels
finalab=np.array([], dtype='uint8')
for i in range (0,len(img_dir)):
	a=globals()[f'label{i}']
	finalab=np.concatenate((finalab,a),axis=0)
#finall = np.concatenate((label0, label1, label2, label3, label4, label5, label6, label7), axis=0)
#finalab=np.array([finall])
finallabel=np.transpose(finalab)

def define_discriminator(in_shape=(224, 224, 3), n_classes=8):  
	in_label = Input(shape=(1,))  # label input
	l = Embedding(n_classes, 1500)(in_label)
	n_nodes = in_shape[0] * in_shape[1]  # 224x224
	l = Dense(n_nodes)(l)
	l = Reshape((in_shape[0], in_shape[1], 1))(l)

	in_image = Input(shape=in_shape)
	concat = Concatenate()([in_image, l])
	dis = Conv2D(128, (4,4), strides=(2,2), padding='same')(concat)
	dis = LeakyReLU(alpha=0.2)(dis)
	dis = Conv2D(128, (4,4), strides=(2,2), padding='same')(dis)
	dis = LeakyReLU(alpha=0.2)(dis)
	dis = Conv2D(128, (4,4), strides=(2,2), padding='same')(dis)
	dis = LeakyReLU(alpha=0.2)(dis)
	dis = Conv2D(128, (4,4), strides=(2,2), padding='same')(dis)
	dis = LeakyReLU(alpha=0.2)(dis)
	dis = Conv2D(128, (4,4), strides=(2,2), padding='same')(dis) 
	dis = LeakyReLU(alpha=0.2)(dis)
	dis = Flatten()(dis)
	dis = Dropout(0.4)(dis)
	out_layer = Dense(1, activation='sigmoid')(dis)

	model=Model([in_image, in_label], out_layer)
	opt = Adam(lr=0.0001, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

test_dis=define_discriminator()
print(test_dis.summary())

def define_generator(latent_dim, n_classes=8):
	in_label = Input(shape=(1,))
	l = Embedding(n_classes, 224)(in_label)
	n_nodes = 7*7
	l = Dense(n_nodes)(l)
	l = Reshape((7, 7, 1))(l)
	in_lat = Input(shape=(latent_dim,))

	n_nodes = 128 * 7 * 7
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((7, 7, 128))(gen)
	concat = Concatenate()([gen, l])
	gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(concat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Conv2DTranspose(128,(4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	
	out_layer = Conv2D(3, (7, 7), activation='tanh', padding='same')(gen)
	model = Model([in_lat, in_label], out_layer)
	return model    
test_gen = define_generator(100, n_classes=6)
print(test_gen.summary())

def load_real_samples():
	trainX = patches
	X=trainX.astype('float32')
	X=(X - 127.5) / 127.5
	trainy = finallabel
	return [X, trainy]

def define_gan(g_model, d_model):
	d_model.trainable = False
    # get noise and label inputs from generator model
	gen_noise, gen_label = g_model.input
    # get image output from the generator model
	gen_output = g_model.output
	gen_output = d_model([gen_output, gen_label])
	model = Model([gen_noise, gen_label], gen_output)

    # compile model Adam
	opt = Adam(lr=0.0002, beta_1=0.9)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

def generate_real_samples(dataset, n_samples):
	images, labels = dataset
	ix = randint(0, images.shape[0], n_samples)
	X, labels = images[ix], labels[ix]
	y = ones((n_samples, 1))  # Label=1 - imgs are real
	return [X, labels], y

def generate_latent_points(latent_dim, n_samples, n_classes=8):
	x_input = randn(latent_dim * n_samples)
	z_input = x_input.reshape(n_samples, latent_dim)
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

def generate_fake_samples(generator, latent_dim, n_samples):
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	images = generator.predict([z_input, labels_input])
	y = zeros((n_samples, 1))  # Label=0 - imgs are fake
	return [images, labels_input], y 


# train the generator and discriminator
disrealloss=[]
disfakeloss=[]
generatorloss=[]
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	for i in range(n_epochs):
		for j in range(bat_per_epo):

         # Train the discriminator on real and fake images, separately (half batch each)
            # Research showed that separate training is more effective. [source: towardsdatascience.com]
           
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)

    # update discriminator model weights
			d_loss_real, _ = d_model.train_on_batch([X_real, labels_real], y_real)
			disrealloss.append(d_loss_real)
            # generate 'fake' examples
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
			d_loss_fake, _ = d_model.train_on_batch([X_fake, labels], y_fake)
			disfakeloss.append(d_loss_fake)
            # prepare points in latent space as input for the generator
			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)

            # create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))

            # update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
			generatorloss.append(g_loss)
		            
# Print losses on this batch
			print('Epoch>%d, Batch%d/%d, d1=%.9f, d2=%.9f g=%.9f' % (i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))
    # save the generator model
	g_model.save("/home/alexandra/home/alexandra/model11.h5")

# Training 

# size of the latent space
latent_dim = 1500
d_model = define_discriminator()
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=300)

# load 
model = load_model("/home/alexandra/home/alexandra/model11.h5")

latent_points, labels = generate_latent_points(1500, 1500)
# specify label - generate latent_dim  images for the labels class

labels=np.array([5]*1500) # generate images
X  = model.predict([latent_points, labels])
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
X = (X*255).astype(np.uint8)
# Plot sample for the generated images 
#def show_plot(examples, n):
#	for i in range(n * n):
#		plt.subplot(n, n, 1 + i)
#		plt.axis('off')
#		plt.imshow(examples[i, :, :, :])
#	plt.show()   
#show_plot(X, 10)

#Save generated images 
for i in range(X.shape[0]):
	img=X[i,:,:,:]
	im=Image.fromarray(img)
	im.save("/home/alexandra/home/alexandra/generatedimages/m11/img{:04d}.tif".format(i))


#save gen and disc loss
import pickle
pickle.dump(disrealloss, open("/home/alexandra/home/alexandra/disrealllossm11.dat","wb"))
pickle.dump(disfakeloss, open("/home/alexandra/home/alexandra/disfakelossm11.dat","wb"))
pickle.dump(generatorloss, open("/home/alexandra/home/alexandra/generatorlossm11.dat","wb"))
