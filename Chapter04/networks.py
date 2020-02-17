def get_generator():
    gen_model = Sequential()
    gen_model.add(Dense(input_dim=100, output_dim=2048))
    gen_model.add(LeakyReLU(alpha=0.2))
    gen_model.add(Dense(256 * 8 * 8))
    gen_model.add(BatchNormalization())
    gen_model.add(LeakyReLU(alpha=0.2))
    gen_model.add(Reshape((8, 8, 256), input_shape=(256 * 8 * 8,)))
    gen_model.add(UpSampling2D(size=(2, 2)))
    gen_model.add(Conv2D(128, (5, 5), padding='same'))
    gen_model.add(LeakyReLU(alpha=0.2))
    gen_model.add(UpSampling2D(size=(2, 2)))
    gen_model.add(Conv2D(64, (5, 5), padding='same'))
    gen_model.add(LeakyReLU(alpha=0.2))
    gen_model.add(UpSampling2D(size=(2, 2)))
    gen_model.add(Conv2D(3, (5, 5), padding='same'))
    gen_model.add(LeakyReLU(alpha=0.2))
    return gen_model

def get_discriminator():
    dis_model = Sequential()
    dis_model.add(Conv2D(128, (5, 5), padding='same', input_shape=(64, 64, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))
    dis_model.add(Conv2D(256, (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))
    dis_model.add(Conv2D(512, (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))
    dis_model.add(Flatten())
    dis_model.add(Dense(1024))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(Dense(1))
    dis_model.add(Activation('sigmoid'))
    return dis_model

#define hyperparameters
dataset_dir = "data/"
batch_size = 128
z_shape = 100
epochs = 10000
dis_learning_rate = 0.0005
gen_learning_rate = 0.0005
dis_momentum = 0.9
gen_momentum = 0.9
dis_nesterov = True
gen_nesterov = True

# Loading images
all_images = []
for index, filename in
enumerate(glob.glob('/Path/to/cropped/images/directory/*.*')):
    image = imread(filename, flatten=False, mode='RGB')
    all_images.append(image)

# Convert to Numpy ndarray
X = np.array(all_images)
X = (X - 127.5) / 127.5

# Define optimizers
dis_optimizer = SGD(lr=dis_learning_rate, momentum=dis_momentum,
nesterov=dis_nesterov)
gen_optimizer = SGD(lr=gen_learning_rate, momentum=gen_momentum,
nesterov=gen_nesterov)

#compile generator model
gen_model = build_generator()
gen_model.compile(loss='binary_crossentropy',
optimizer=gen_optimizer)

#compile discriminator model
dis_model = build_discriminator()
dis_model.compile(loss='binary_crossentropy',
optimizer=dis_optimizer)

#build and compile adversarial model
adversarial_model = Sequential()
adversarial_model.add(gen_model)
dis_model.trainable = False
adversarial_model.add(dis_model)
adversarial_model.compile(loss='binary_crossentropy',
optimizer=gen_optimizer)

#initialize tensorboard
tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()),
write_images=True, write_grads=True, write_graph=True)
tensorboard.set_model(gen_model)
tensorboard.set_model(dis_model)

#need to figure out beyone here
for epoch in range(epochs):
    print("Epoch is", epoch)
    number_of_batches = int(X.shape[0] / batch_size)
    print("Number of batches", number_of_batches)
    for index in range(number_of_batches):

z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
image_batch = X[index * batch_size:(index + 1) * batch_size]
generated_images = gen_model.predict_on_batch(z_noise)
y_real = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
y_fake = np.random.random_sample(batch_size) * 0.2
dis_loss_real = dis_model.train_on_batch(image_batch, y_real)
dis_loss_fake = dis_model.train_on_batch(generated_images, y_fake)
d_loss = (dis_loss_real + dis_loss_fake)/2
print("d_loss:", d_loss)

z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
g_loss = adversarial_model.train_on_batch(z_noise, [1] * batch_size)
print("g_loss:", g_loss)
