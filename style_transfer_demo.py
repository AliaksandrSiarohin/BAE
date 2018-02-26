from style_transfer_model import style_transfer_model, deprocess_input, preprocess_input
import numpy as np

model = style_transfer_model()

from skimage.io import imread
style = imread('water_color.jpg')
image = imread('cornell_cropped.jpg')


# xx, yy = np.meshgrid(range(image.shape[1]), range(image.shape[0]))
# print (xx.shape)
# print (yy.shape)
# z1 = np.sin(xx / 8.0) + np.sin(xx / 25.0) + np.sin(xx / 39.0)
# z2 = np.sin(yy / 15.0) + np.sin(xx / 31.0) + np.sin(xx / 17.0)
# z3 = np.cos(xx / 22.0)  + np.cos(xx / 29.0)  + np.cos(xx / 7.0)
#
# image = np.concatenate([z1[..., np.newaxis], z2[..., np.newaxis], z3[..., np.newaxis]], axis=-1)
# print (image.shape)
import pylab as plt
plt.imshow(image)
plt.show()

#image = np.random.uniform(size=image.shape) * 255

style = preprocess_input(style)
image = preprocess_input(image)
res = model.predict([image, style])
res = deprocess_input(res)
import pylab as plt
plt.imshow(res)
plt.show()


