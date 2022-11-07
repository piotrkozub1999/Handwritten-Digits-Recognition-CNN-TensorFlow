import tensorflow as tf
from keras import models
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
# from keras.layers import Dropout, LeakyReLU, BatchNormalization, Activation, Input
# from keras.optimizers import SGD

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  # ladowanie danych z bazy mnist

train_images = train_images.reshape((60000, 28, 28, 1))  # zmiana rozmiaru obrazów na 28x28
train_images = train_images.astype('float32') / 255  # przeskalowanie do zakresu [0,1], ustawienie precyzji 32-bitowej
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = tf.keras.utils.to_categorical(train_labels)  # konwersja danych integer na kategoryczne(one-hot encoding), żeby keras mógł na nich działać, skalujemy dane
test_labels = tf.keras.utils.to_categorical(test_labels)  # np. 7 to [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]

### Prosta sieć jednokierunkowa ###
# model = models.Sequential()
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=5, batch_size=64)


### Sieć konwolucyjna ###
model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Model z jedną warstwą Conv2D
""" model = models.Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
opt = SGD(learning_rate=0.01, momentum=0.9)
model.summary()
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy']) 
model.fit(train_images, train_labels, epochs=5, batch_size=64) """

# Model z Dropout
""" model = models.Sequential() 
input_shape=(28,28,1)
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
                              input_shape=input_shape))
model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy']) 
model.fit(train_images, train_labels, epochs=5, batch_size=64) """

# Model z Dropout, Batch Normalization, LeakyReLU
""" model = models.Sequential() 
model.add(Conv2D(128,(3,3),input_shape=(28,28,1),activation='relu',padding='same'))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(MaxPooling2D(pool_size=(2,2),padding='same')) 

model.add(Conv2D(256,(3,3),padding='same',activation='relu')) 
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(128,(3,3),padding='same',activation='relu')) 
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),padding='same',activation='relu')) 
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),padding='same',activation='relu')) 
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.3))

model.add(Flatten())  
model.add(Dense(64,activation='relu')) 
model.add(Dense(10,activation='softmax'))

model.summary()
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy']) 
model.fit(train_images, train_labels, epochs=5, batch_size=64) """

# Model z ReLU
""" model = models.Sequential() 
model.add(Conv2D(64, kernel_size = (3,3), strides = (1,1), data_format = 'channels_last', input_shape = (28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(128, kernel_size = (3,3)))
model.add(Dropout(0.4))
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size = (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.4))
model.add(Activation('relu'))
model.add(Dense(10, activation = 'softmax'))
model.summary()
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64) """

# Model z dwiema warstwami Conv2D i Dropout
""" model = models.Sequential() 
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)  """

# Model funkcjonalny
"""inputs = Input(shape=(28, 28, 1))
inputs.shape
conv2D = Conv2D(24, (3, 3), activation='relu', kernel_initializer='he_uniform')(inputs)
Maxpool = MaxPooling2D((2, 2))(conv2D)
flatten = Flatten()(Maxpool)
dense = Dense(512, activation='relu', kernel_initializer='he_uniform')(flatten)
output = Dense(10, activation='softmax')(dense)
opt = SGD(learning_rate=0.01, momentum=0.9)
model = models.Functional(inputs=inputs, outputs=output, name="mnist_model") 
model.summary()
model.compile(optimizer=opt, 
              loss='categorical_crossentropy', 
              metrics=['accuracy']) 
model.fit(train_images, train_labels, epochs=5, batch_size=64) """

######## Sprawdzamy skuteczność ###########
score = model.evaluate(test_images, test_labels)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

######## Zapisywanie modelu sieci ###########
model.save('mnist.h5')
print("Saving the model as mnist.h5")
