import numpy as np
import cv2
import h5py
from segmentation_models import Unet
from segmentation_models import get_preprocessing
import glob
import random
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

#-------------------------------------#

BACKBONE = 'resnet50'
preprocess_input = get_preprocessing(BACKBONE)

images = sorted(glob.glob(f"../2depth/data/synthetic/*/*/*.h5"))
labels = images
reals = sorted(glob.glob(f'../2depth/reals/*.png'))
depthpath = sorted(glob.glob(f'../2depth/results/synthetic_mask=False_arch=resnet_bs=8_lr=0.0001_ep=2_br=1_sd=1/demo_results/*_output.npy'))

print("# of images:"+str(len(images)))

x = []
y = []
classes = 2 
ratio = 0.8
input_shape = (256, 192)

for img_path in images:
  with h5py.File(img_path) as f:
    img = np.array(f["rgb"])
    img = img.transpose(1, 2, 0)
    img = cv2.resize(img, input_shape)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[:,:,(0)] = img[:,:,(0)] + random.uniform(170, 190)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    noise=np.random.normal(0,random.uniform(0, 15), np.shape(img))
    img=img+np.floor(noise)
    img[img>255]=255
    img[img<0]=0

    img = np.array(img, dtype=np.float32)
    img *= 1./255
    x.append(img)

for label_path in labels:
  with h5py.File(label_path) as f:
    label = np.array(f["mask"], np.float32)
    label = cv2.resize(label, input_shape)
    img = []
    for label_index in range(classes):
      img.append(label == label_index)
    img = np.array(img, dtype=np.float32)
    img = img.transpose(1, 2, 0)
    y.append(img)

x = np.array(x)
y = np.array(y)
x = preprocess_input(x)

np.random.seed(1)
np.random.shuffle(x)
np.random.seed(1)
np.random.shuffle(y)

p = int(ratio * len(x))
x_train = x[:p]
y_train = y[:p]
x_val = x[p:]
y_val = y[p:]

#-------------------------------------#

data_gen_args = dict(
                  #  rotation_range=20,
                  #  horizontal_flip = True,
                  #  width_shift_range=0.1,
                  #  height_shift_range=0.1,
                  #  zoom_range=0.2
)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1
image_datagen.fit(x_train, augment=True, seed=seed)
mask_datagen.fit(y_train, augment=True, seed=seed)

image_generator = image_datagen.flow(
    x_train,
    # class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow(
    y_train,
    # class_mode=None,
    seed=seed)

train_generator = zip(image_generator, mask_generator)

#-------------------------------------#

model = Unet(BACKBONE, classes=classes, encoder_weights=None, activation='sigmoid')
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

#ifは動作確認用.elseは論文の実験での値
if(len(images) < 5):
  step = 2
  eps = 2
else:
  step = 50
  eps = 200

history = model.fit(
    train_generator,
    steps_per_epoch=step,
    epochs=eps,
    validation_data=(x_val, y_val)
)
#-------------------------------------#
depth = []

for i in range (len(depthpath)):
  depth.append(np.load(depthpath[i]))
  depth[i] = cv2.resize(depth[i], dsize=(256, 192))

#-------------------------------------#

i = 0
for real in reals:
  real = cv2.imread(real) 
  alpha = cv2.cvtColor(real, cv2.COLOR_BGR2BGRA)
  real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB) #BGR -> RGB
  real = real / 255
  real = cv2.resize(real, dsize=(256, 192))
  preds = model.predict(real[np.newaxis, ...])
  pred_img = np.argmax(preds[0], axis=2)
  alpha = alpha / 255
  alpha = cv2.resize(alpha, dsize=(256, 192))
  alpha[:,:,3] = pred_img * (1-depth[i])
  alpha = alpha*255
  alpha = cv2.resize(alpha, dsize=(137, 137))
  cv2.imwrite("./results/{:0=3}".format(i)+".png", alpha)
  i += 1


