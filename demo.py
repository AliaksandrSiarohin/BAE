from style_transfer_model import style_transfer_model, deprocess_input, preprocess_input

model = style_transfer_model()

from skimage.io import imread
style = imread('style.jpg')
image = imread('content.jpg')


style = preprocess_input(style)
image = preprocess_input(image)
res = model.predict([image, style])
res = deprocess_input(res)

import pylab as plt
plt.imshow(res)
plt.show()


