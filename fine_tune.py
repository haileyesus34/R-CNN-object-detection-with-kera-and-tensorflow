from imagesearch import config 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications import MobileNetV2 
from tensorflow.keras.layers import AveragePooling2D 
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Input 
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from imutils import paths 
import matplotlib.pyplot as plt 
import numpy as np 
import pickle 
import os 

init_lr = 1e-4
epochs = 5
bs = 32

imagePaths = list(paths.list_images(config.base_path))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = load_img(imagePath, target_size=config.input_dims)
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)


data = np.array(data, dtype='float32')
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

aug = ImageDataGenerator(
    rotation_range=20, 
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

baseModel = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(128, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation='softmax')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers: 
    layer.trainable = False

opt = Adam(init_lr)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

H = model.fit(
    aug.flow(x_train, y_train, batch_size=bs),
    steps_per_epoch = len(x_train) // bs,
    validation_data = (x_test, y_test),
    epochs=epochs
)
predIdxs = model.predict(x_test, batch_size=bs)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(y_test.argmax(axis=1), predIdxs, target_names=lb.classes_))

model.save(config.model_path)

f = open(config.encoder_path, 'wb')
f.write(pickle.dumps(lb))
f.close()

N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')

