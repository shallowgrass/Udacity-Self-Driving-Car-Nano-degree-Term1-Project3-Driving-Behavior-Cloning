# This script trains a keras model for human driver behavioral cloning.
# Input: camera image, Output: steering wheel angle.
# April 22nd, 2018, By S.CHEN

# Imports
import numpy as np
import os, cv2, csv, keras, sklearn
from datetime import datetime

class SetParams(object):
   '''
   Group all (hyper)parameters.
   '''
   def __init__(self):
      # data params
      self.logfile = './data/driving_log.csv'
      self.input_shape = (160, 320, 3) 
      self.crop_region = ((60,25), (0,0))
      self.use_augmented_data = True
      
      # training parmas
      self.training_flag = True
      self.batch_size = 256
      self.optimizer = 'adam'
      self.learning_rate = 0.01
      self.lr_decay_step = 20
      self.regularizer_val = 0.001
      self.dropout_rate = 0.1
      self.num_epochs = 100

      # checkpoint params
      self.weights_dir = './weights'
      self.continued_training = False
      self.ckpt_file = './weights/ckpt.h5'

class LoadData(object):
   '''
   Load train(dev) data
   '''
   def __init__(self, params):
      self.logfile = params.logfile
      self.samples = []
      self.train_samples = []
      self.validation_samples = []
      self.use_augmented_data = params.use_augmented_data
      
      # Get training data from log file
      self.get_samples()

      # Split train/dev set
      self.split_data()
      
      # Compute number of batches
      self.num_train_batches      = len(self.train_samples)      // params.batch_size
      self.num_validation_batches = len(self.validation_samples) // params.batch_size
      
      # Compute standard variation of steering angle (to be used as correction value)
      self.compute_angle_correction()

   def get_samples(self):
      '''Get data from log file'''
      with open(self.logfile) as f:
         reader = csv.reader(f)
         next(reader)    # Skip header
         for line in reader:
            self.samples.append(line)

   def compute_angle_correction(self):
      '''Compute std value of steering angles, and use it as correction angle'''
      angles = []
      with open(self.logfile) as f:
         reader = csv.reader(f)
         next(reader)    # Skip header
         for line in reader:
            angles.append(float(line[3]))
      correction = np.std(angles)
      self.angle_correction = max(correction, 0.01)
      print('Angle correction used in data augmentation is {:.4f}'.format(self.angle_correction))
      
   def split_data(self):
      '''
      Split data set to train/dev set
      '''
      from sklearn.model_selection import train_test_split
      self.train_samples, self.validation_samples = train_test_split(self.samples, test_size=0.2)

   def data_augmentation(self, batch_sample, train_flag):
      '''
      Apply data augmentation on input image, and change target value accordingly
      '''
      img_dir = 'data/IMG/'
      splitter = '/'
      # When no augmented data are to be used, load the central image only
      if (train_flag == False) or (self.use_augmented_data == False):
         name = img_dir + batch_sample[0].split(splitter)[-1]
         img = cv2.imread(name)
         angle = float(batch_sample[3])
      else:  # In case of using augmented data:
         #print('correction angle: {:.4f}'.format(self.angle_correction))
         angle = float(batch_sample[3])            # Get the steering angle
         augmentation_ratio = 0.5                  # Set augmentation ratio
         if(np.random.uniform() > 0.5):
            img_name = img_dir + batch_sample[0].split(splitter)[-1]
            img = cv2.imread(img_name)
            if (np.random.uniform() > 0.5):        # Half of time use its flipped version
               img = cv2.flip(img, flipCode=1)
               angle = -angle
         elif(np.random.uniform() > 0.5):          # Use left image
            img_name = img_dir + batch_sample[1].split(splitter)[-1]
            img = cv2.imread(img_name)
            angle += self.angle_correction         # apply angle correction
         else:                                     # Use right image
            img_name = img_dir + batch_sample[2].split(splitter)[-1]
            img = cv2.imread(img_name)
            angle -= self.angle_correction         # apply angle correction
      img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB
      return img, angle

   def generator(self, samples, batch_size, train_flag):
      '''Generate mini-batch data for training'''
      num_samples = len(samples)
      while True:
         np.random.shuffle(samples)   # Shuffle data for each epoch
         for offset in range(0, num_samples, batch_size): # Yield mini-batchs
            batch_samples = self.samples[offset:offset+batch_size]
            images = []
            angles = []
            for sample in batch_samples:
               # Perform data augmentation on training samples
               img, angle = self.data_augmentation(sample, train_flag) 
               images.append(img)
               angles.append(angle)
            X = np.array(images)
            y = np.array(angles)
            yield sklearn.utils.shuffle(X, y)
                  
def behavior_clone_model(params):
   '''
   Create keras model to predict steering angle given camera image
   '''
   from   keras.models               import Sequential
   from   keras.layers               import Flatten, Dense, Dropout, Activation, Input, Lambda, Cropping2D
   from   keras.layers.convolutional import Conv2D, MaxPooling2D
   from   keras.layers.normalization import BatchNormalization
   from   keras                      import regularizers
   from   keras.layers               import Lambda
   model = Sequential()
   
   # Crop out RoI
   model.add(Cropping2D(cropping=params.crop_region, input_shape=params.input_shape))
   
   # Whitening input
   model.add(Lambda(lambda x: x / 255.0 - 0.5))
   
   # conv1 --> batch_norm1 --> relu1 --> pool1
   # Stride more along horizental axis to compensate vertical depth compression
   model.add(Conv2D(24, (5,5), strides=(2,4), kernel_regularizer = regularizers.l1_l2(params.regularizer_val)))
   model.add(BatchNormalization())
   model.add(Activation('relu', name='relu_1'))
   model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool_1'))
   
   # conv2 --> batch_norm2 --> relu2 --> pool2
   model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer = regularizers.l1_l2(params.regularizer_val)))
   model.add(BatchNormalization())
   model.add(Activation('relu', name='relu_2'))
   model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool_2')) 

   # conv3 --> batch_norm3 --> relu3 --> pool3
   model.add(Conv2D(48, (3,3), padding='same', kernel_regularizer = regularizers.l1_l2(params.regularizer_val)))
   model.add(BatchNormalization())
   model.add(Activation('relu', name='relu_3'))
   model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool_3')) 

   # conv4, convolve activation map into vector (to conect 1x1 convlution)
   model.add(Conv2D(64, (3,9), kernel_regularizer = regularizers.l1_l2(params.regularizer_val)))
   model.add(BatchNormalization())
   model.add(Activation('relu', name='relu_4'))
   
   # Dropout 4
   model.add(Dropout(params.dropout_rate, name='dropout_4'))

   # Conv5
   model.add(Conv2D(16, (1,1), kernel_regularizer = regularizers.l1_l2(params.regularizer_val)))
   model.add(BatchNormalization())
   model.add(Activation('relu', name='relu_6'))

   # Conv6: output steering angle value
   model.add(Conv2D(1, (1,1), kernel_regularizer = regularizers.l1_l2(params.regularizer_val)))
   
   # Reshape to scalar
   model.add(keras.layers.Reshape((1,)))

   # View graph layout
   model.summary()
   return model

def compile_model(model, params):
   '''
   Set training optimizer and compile model
   '''
   opt = keras.optimizers.Adam(lr=params.learning_rate)
   model.compile(loss='mse', optimizer=opt)

def show_training(train_history):
   '''
   Visualize training and validation process
   '''
   import matplotlib.pyplot as plt
   print(train_history.history.keys())
   plt.plot(train_history.history['loss'])
   plt.plot(train_history.history['val_loss'])
   plt.title('model mean squared error loss')
   plt.ylabel('mean squared error loss')
   plt.xlabel('epoch')
   plt.legend(['training set', 'validation set'], loc='upper right')
   plt.show()
   plt.savefig('training_process.png', bbox_inches='tight')
   
def train_model(model, train_data, params):
   '''
   Train/validate model on training/validation data, and save model 
   '''
   # Generator for train/dev set
   train_generator      = train_data.generator(train_data.train_samples,      
                                               params.batch_size, 
                                               True)
   validation_generator = train_data.generator(train_data.validation_samples, 
                                               params.batch_size, 
                                               False) # Set train_flag to False, which disables data augmentation.

   # Prepare to save
   if not os.path.isdir(params.weights_dir): os.mkdir(params.weights_dir)
   
   # Load checkpoint file if in continued training
   if params.continued_training == True:
      assert os.path.exists(params.ckpt_file)
      model.load_weights(params.ckpt_file, by_name=True)

   # Define learning rate schedule
   # Linear lr decay
   lr_schedule = lambda x : params.learning_rate / (10.**(x//params.lr_decay_step))
   ## Ploy lr decay 
   #lr_schedule = lambda x : params.learning_rate * (1.0-x/params.num_epochs)**0.9
   lr_schedule_callback = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
   ckpt_name = params.weights_dir + '/' + '{val_loss:.4f}_{epoch:03d}.h5'
   save_model_callback = keras.callbacks.ModelCheckpoint(ckpt_name,
                                                         monitor='val_loss', 
                                                         verbose=0, 
                                                         save_best_only=False, 
                                                         save_weights_only=False, 
                                                         mode='auto', 
                                                         period=1)
   callbacks = [lr_schedule_callback, save_model_callback]

   # Train the model
   if params.training_flag == True:
      train_history = model.fit_generator(train_generator,
                                          steps_per_epoch  = train_data.num_train_batches,
	   				                        validation_data  = validation_generator, 
	   				                        validation_steps = train_data.num_validation_batches,
	   				                        epochs           = params.num_epochs, 
	   				                        verbose          = 1,
                                          callbacks        = callbacks)

      # Visualization training process
      show_training(train_history)
   
   # Save trained model
   model.save('{}.h5'.format(datetime.now().strftime("%m_%d_%H_%M")))
   
def main():
   # Set all training parameters
   params = SetParams()

   # Load training data
   train_data = LoadData(params)
   
   # Create model
   model = behavior_clone_model(params)

   # Compile model
   compile_model(model, params)

   # Train/validate and save model
   train_model(model, train_data, params)

if __name__ == '__main__':
   main()