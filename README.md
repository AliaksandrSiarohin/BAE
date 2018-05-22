#

[Check out our paper]()

<style>
.grid-container {
  display: grid;
  grid-template-columns: auto auto;
}
.grid-item {
  text-align: center;
}
</style>

<div class="grid-container">
  <div class="grid-item"><img src="sup-mat/cornell_scary.jpg"/> <figcaption align="center">scary</figcaption> </div>
  <div class="grid-item"><img src="sup-mat/cornell_gloomy.jpg"/> <figcaption align="center">glomy</figcaption> </div>
  <div class="grid-item"><img src="sup-mat/cornell_scary.jpg"/> <figcaption align="center">scary</figcaption> </div>
  <div class="grid-item"><img src="sup-mat/cornell_scary.jpg"/> <figcaption align="center">scary</figcaption> </div>
</div>



### Requirments
* python2
* numpy
* scipy
* skimage
* pandas
* tensorflow
* keras
* tqdm 


### Demo

1. Download pretrained [models](https://yadi.sk/d/PXSo4UkN3WN35P).

2. Launch

```python style_optimization_demo.py --image_shape 3,512,512 --adaptive_grad 1\
--weight_image 10 --lr_decay 0.9 --score_type gloomy --number_of_iters 100\
--alpha_sigma 0.25 --output_dir output/cornell_gloomy --content_image cornell_cropped.jpg```

Check ```cmp.py``` for explanation of the commands.

 ### Dataset


 ### Evaluation


 ### Training




