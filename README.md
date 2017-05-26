# CycleGAN in TensorFlow

TensorFlow implementation of [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf).

Torch implementation can be found [here](https://github.com/junyanz/CycleGAN)  
PyTorch implementation can be found [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

### Prerequisites  
* Tensorflow  
* numpy  
* scipy  

## Downloading datasets  
```bash
bash ./datasets/download_cyclegan_dataset.sh dataset_name  
```
Available datasets:  
* __facades__: 400 images from the CMP Facades dataset.
* __cityscapes__: 2975 images from the Cityscapes training set.  
* __maps__: 1096 training images scraped from Google Maps.  
* __horse2zebra__: 939 horse images and 1177 zebra images downloaded from ImageNet using keywords wild horse and zebra.  
* __apple2orange__: 996 apple images and 1020 orange images downloaded from ImageNet using keywords apple and navel orange.  
* __summer2winter_yosemite__: 1273 summer Yosemite images and 854 winter Yosemite images were downloaded using Flickr API. See more details in our paper.  
* __monet2photo__ , __vangogh2photo__, __ukiyoe2photo__ , __cezanne2photo__: The art images were downloaded from Wikiart. The real photos are downloaded from Flickr using the combination of the tags landscape and landscapephotography. The training set size of each class is Monet:1074, Cezanne:584, Van Gogh:401, Ukiyo-e:1433, Photographs:6853.  
* __iphone2dslr_flower__: both classes of images were downlaoded from Flickr. The training set size of each class is iPhone:1813, DSLR:3316. See more details in the paper.  

## Training
```bash
python main.py --is_train=True --data=apple2orange
```
 

## Testing
```bash
python main.py --is_train=False --data=apple2orange
```

Running this code generates images in 2 folders.  
  
__fake_A__ folder contains fake images generated from domain B  
__fake_B__ folder contains fake images generated from domain A  

## Tensorboard  
```bash
tensorboard --logdir=./graphs
```

Default port set to 6006.  
This will enable you to visualize training details.  

## Results  

Results from __'maps'__ dataset:  
![maps_1](./checkpoints/images(maps)/model_015000.jpg)
![maps_2](./checkpoints/images(maps)/model_024000.jpg)  
![maps_3](./checkpoints/images(maps)/model_034000.jpg)
![maps_4](./checkpoints/images(maps)/model_037000.jpg)  

Results from __'apple2orange'__ dataset:  
![a2o_1](./checkpoints/images(apple2orange)/model_030000.jpg)
![a2o_2](./checkpoints/images(apple2orange)/model_031000.jpg)  
![a2o_3](./checkpoints/images(apple2orange)/model_069000.jpg)
![a2o_4](./checkpoints/images(apple2orange)/model_084000.jpg)  

## Author

Seung Shin / [@shinseung428](http://shinseung428.github.io)