import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from keras.layers import Dense, Activation, BatchNormalization, Input, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.models import Sequential, Model


img_shape = (28,28,1)

def build_generator():
    #Input - Noise
    #Output - Fake Image with label True to fake discriminator
    
    noise_shape = (100,)
    
    model = Sequential()
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation="tanh"))
    model.add(Reshape(img_shape))
    
    model.summary()
    
    noise = Input(shape=noise_shape)
    img = model(noise)
    
    return Model(noise,img)

def build_discriminator():
    #Input - Image
    #Output - Prediction of Image True/False
    
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation="sigmoid"))
    
    img = Input(shape=img_shape)
    validity = model(img)
    
    return Model(img, validity)

def train(epochs, batch_size=64, save_interval=50):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255
    x_train = np.expand_dims(x_train, -1)
    
    half_batch = int(batch_size/2)
    
    for epoch in range(epochs):
        idx = np.random.randint(0,x_train.shape[0],half_batch)
        imgs = x_train[idx]
        
        noise = np.random.normal(0, 1, (half_batch, 100))
        
        gen_imgs = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) 
        
        noise = np.random.normal(0, 1, (batch_size, 100)) 
        
        valid_y = np.array([1] * batch_size)
        
        g_loss = combined.train_on_batch(noise, valid_y)
        
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        
        if epoch % save_interval == 0:
            save_imgs(epoch)
            
def save_imgs(epoch):
    image_file = "./images"
    if not os.path.exists(image_file):
        os.mkdir(image_file)
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/mnist_%d.png" % epoch)
    plt.close()
    
optimizer = Adam(0.0002, 0.5)
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

z = Input(shape=(100,))
img = generator(z)
discriminator.trainable = False  
valid = discriminator(img)

combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)


train(epochs=50000, batch_size=64, save_interval=2500)

generator.save('generator_model.h5')
