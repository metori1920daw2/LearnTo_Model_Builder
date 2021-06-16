

import tensorflow as tf
from ML.model import ResNet
from Tools.utils import lr_schedule
from Tools.utils import existsfolder
from ML.Dataset import load_dataset
from Tools.utils import save_model_tflite
from Tools.utils import Create_metadata_model
from Tools.utils import SaveReport
from Tools.utils import visualize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from imutils import build_montages
import numpy as np
import cv2
from datetime import datetime
now = datetime.now().strftime("%Y%m%d_%H%M%S")



(trainX, testX, trainY, testY,classWeight) = load_dataset('az_dataset.csv')



batch_size = 32  
epochs = 20
num_classes = 36
#input_shape = (32, 32, 3)
input_shape = (32, 32, 1)
depth = 32

print("Build Resnet network!") 
model = ResNet(input_shape = input_shape, depth=depth,num_classes=num_classes)
model.compile(loss ='categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(amsgrad=True, decay=0.001/epochs), metrics =['accuracy'])

#checkpoint_filepath = '.\\assets\\checkpoint'
#existsfolder(checkpoint_filepath)
#model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_filepath,save_weights_only=True, monitor='val_accuracy', mode='max',save_best_only=True)
#checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = save_dir, monitor ='val_acc', verbose = 1, save_best_only = True)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
  
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor = np.sqrt(0.1),cooldown = 0, patience = 5, min_lr = 0.5e-6)

callbacks = [lr_reducer, lr_scheduler]

aug = ImageDataGenerator(rotation_range=10,zoom_range=0.05,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.15,horizontal_flip=False,fill_mode="nearest")

print("Iniciar entrenament!")
history = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),validation_data=(testX, testY),steps_per_epoch=len(trainX)//batch_size,epochs=epochs,callbacks = callbacks, class_weight=classWeight,verbose=1)
print("Final d'entrebament traning!")
exporPath = './assets/Models_{0}'.format(now)
print("Crear gràfica  de rediment!")
existsfolder(exporPath)

visualize(history=history.history,Path=exporPath)
print("save Model with metadata!")
save_model_tflite(model)


#Create_metadata_model(exporPath,"chars.tflite","CharsLlabels.txt")
labels = [l for l in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"]


print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=batch_size)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=labels))
SaveReport(testY.argmax(axis=1),predictions.argmax(axis=1), labels=labels,Path=exporPath)

model.save('{0}_model.h5'.format(now))

images = []


for i in np.random.choice(np.arange(0, len(testY)), size=(49,)):
	probs = model.predict(testX[np.newaxis, i])
	prediction = probs.argmax(axis=1)
	label = labels[prediction[0]]

	image = (testX[i] * 255).astype("uint8")
	color = (0, 255, 0)


	if prediction[0] != np.argmax(testY[i]):
		color = (0, 0, 255)

	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,color, 2)

	images.append(image)


montage = build_montages(images, (96, 96), (7, 7))[0]


cv2.imshow("DeCoder", montage)
cv2.waitKey(0)