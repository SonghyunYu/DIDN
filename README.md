# DIDN
Pytorch Implementation of "**Deep Iterative Down-Up CNN for Image Denoising**". [[PDF]](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Yu_Deep_Iterative_Down-Up_CNN_for_Image_Denoising_CVPRW_2019_paper.pdf)

Our method took **second** and **third** place at each track in [[NTIRE 2019 Challenge on Real Image Denoising]](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Abdelhamed_NTIRE_2019_Challenge_on_Real_Image_Denoising_Methods_and_Results_CVPRW_2019_paper.pdf)

If you use our work in your research or publication, please cite our paper:
```
@inproceedings{yu2019deep,
  title={Deep iterative down-up CNN for image denoising},
  author={Yu, Songhyun and Park, Bumjun and Jeong, Jechang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2019}
```
## Environment  
  python 3.6   
  pytorch 1.0.0  
  MATLAB (for pre-processing)  

## Prepare data
You need to prepare data befor training / testing the model.    
Please use matlab files in './data' folder.   
For training, we used [DIV2K](http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf) dataset, and only sample images are included in this project.  
Please download the whole dataset for training the model.

## Test
Download Pre-trained models:[[google drive]](https://drive.google.com/open?id=1dwc6T4Kk5tjwiSFuOiogoWts55GsbMeC)  
for grayscale images, gray_eval.py  
for color images, color_eval.py  
```
optional arguments:  
  --cuda : using cuda   
  --model : model path  
  --output_path : output image path   
  --self_ensemble : using self-ensemble  
  --gpus : gpu ids  
  
  You can also change test dataset in the code!
```
Example:  
```
python gray_eval.py --cuda --self_ensemble
```

## Training
```
optional arguments:  
  --batchSize : batchsize (default: 16)  
  --nEpochs : number of epochs (default: 50)  
  --lr : initial learning rate (default: 0.0001)  
  --cuda : using cuda   
  --resume : path to checkpoint (default: none)  
  --start_epoch : manual starting epoch number (useful for resume, default: 1)  
  --threads : number of threads for data loader (default: 0)  
  --gpus : gpu ids  
  
  You can also change training data path and validation data path in the code!  
```
Example:  
```
python gray_train.py --cuda --threads 8
```

## Weight ensemble
For weight ensemble, train the model multiple times manually using --resume option,  
and averaging weights using 'weight_ensemble_2.py'

  
## Contact
If you have any questions about the code or paper, please contact fkdlzmtld@gmail.com
