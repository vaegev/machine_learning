from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# initialize CNN
classifier = Sequential()

# step 1 adding Convolution layer
classifier.add(Conv2D(filters=32, 
                      kernel_size=(3, 3),
                      input_shape=(64, 64, 3), 
                      activation='relu'))

# step 2 Pooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# step 3 flattning
classifier.add(Flatten())

# step 4 full connection
classifier.add(Dense(units=128, activation='relu'))

# step5 output layer
classifier.add(Dense(units=1, activation='sigmoid'))

# compilling CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fitting the CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(train_set,
                        steps_per_epoch=8000,
                        epochs=1,
                        validation_data=test_set,
                        validation_steps=2000)
