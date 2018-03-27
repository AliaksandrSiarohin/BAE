from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import keras
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset_train", default='dataset/emotion_scary/internal', type=str)
parser.add_argument("--dataset_val", default='dataset/emotion_scary/external', type=str)
parser.add_argument("--checkpoint_name", default="models/scary_internal.h5", type=str)
parser.add_argument("--images_per_epoch", default=5000, type=int)
parser.add_argument("--batch_size", default=24, type=int)
parser.add_argument("--epochs", default=50, type=int)
args = parser.parse_args()

base_model = InceptionV3(weights='imagenet', include_top=False)

img_width, img_height = 299, 299

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

predictions = Dense(2, activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True


model.compile(optimizer=keras.optimizers.SGD(lr=1e-3, decay=1e-6, nesterov=True),
              loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    args.dataset_train,
    target_size=(img_height, img_width),
    batch_size=args.batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    args.dataset_val,
    target_size=(img_height, img_width),
    batch_size=args.batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=args.images_per_epoch // args.batch_size,
    epochs=args.epochs,
    validation_data=validation_generator,
    validation_steps=args.images_per_epoch // args.batch_size)

model.save(args.checkpoint_name)
