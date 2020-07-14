# Through the Looking Glass: <br> Neural 3D Reconstruction of Transparent Shapes <br> ([Project page](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20Transparent/) )

## Useful links
* Project page: http://cseweb.ucsd.edu/~viscomp/projects/CVPR20Transparent/
* Dataset creation: https://github.com/lzqsd/TransparentShapeDatasetCreation
* Renderer: https://github.com/lzqsd/OptixRenderer
* Trained models: coming soon
* Real data: coming soon
* Real data processing: coming soon
* Synthetic dataset: coming soon

## Overview
This is the official code release of our paper [Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes, CVPR 2020](https://arxiv.org/abs/2004.10904). Please consider citing this paper if you find this code useful in your project. Please contact us if you have any questions or issues. 
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20Transparent/github/TransShape.gif)

## Prerequisite
In order to run the code, please install
* Pytorch: versions later than 1.0 might be enough
* Colmap: Please install Colmap from this [link](https://colmap.github.io/). We use Colmap to reconstruct mesh from point cloud prediction. 
* Meshlab: Please install Meshlab from this [link](https://www.meshlab.net/). We use the subdivision algorithm in Meshlab to smooth the surface so that there is no artifacts when rendering transparent shape. This is important when the BRDF is a delta function. 

## Instructions
The code have 2 parts. The normal prediction part is included in `Normal` directory and the point cloud prediction part is in `PointCloud` directory. We will use 10-view reconstruction as an example to demonstrate how to use the code. The instructions to train the network for 5-view and 20 view reconstructions are listed after.
1. Prepare the dataset.
    * Please visit this [link](https://github.com/lzqsd/TransparentShapeDatasetCreation) to check how to create the transparent shape dataset. Our rendered dataset will also be released soon. However, you may not be able to directly test on our dataset since the paths towards environment maps cannot be the same. Please save the shapes and images in the directory `../Data/Shapes` and `../Data/Images10` respectively.
2. Go to `Normal` directory, run `python train.py --cuda --isAddCostVolume --camNum 10`
   * This will start training the network for the two-bounce normal prediction. Some useful flags are listed below.
      1. `--isAddCostVolume`： whether to use cost volume in normal prediction
      2. `--poolingMode`: Control how we process the feature extracted from cost volume. By default we will use the learnable pooling. 
      3. `--isNoErrMap`: whether to add rendering error as an input for normal prediction. (line 248)
3. Run `python test.py --cuda --isAddCostVolume --camNum 10` to test the trained model. 
4. Run `python optimize.py --cuda --isAddCostVolume --camNum 10 --mode train`.
   * It will use rendering error to optimize the latent vector of the decoder network for better normal prediction. The optimized predicted normals will be saved in the image directory for point cloud prediction. 
5. Run `python optimize.py --cuda --isAddCostVolume --camNum 10 --mode test`
6. Go to `PointCloud` directory, run `python sampleGTPointUniformly.py --mode train` and `python sampleGTPointUniformly.py --mode test`.
   * It will sample uniformly on the ground-truth shape. The sampled results will be saved in `.npy` file in the shape directory. 
7. Run `python sampleVisualHullPoint.py --mode train` and `python sampleVisualHullPoint.py --mode test`
   * It will first sample points uniformly on visual hull geometry and the find their nearest neighbor points on the ground-truth geometry. The results will be saved in `.npy` files in the shape directory. 
8. Run `python trainPoint.py --camNum 10`
   * This will start training the customized PointNet++ for shape reconstruction. Some useful flags are listed below.
      1. `--viewMode`: Control how we choose the view when generating features for each point. The by default choice is to choose the view with the lowest rendering error. Please read our paper for more details.
      2. `--lossMode`: Control the loss to train the network. The by default choice is to use Chamfer loss, which leads to the best results. Please read our paper for more details. 
      3. We also offer different variations of PointNet++ to help reproduce the ablation studies in the supplementary material (`model maxPooling`, `model_noNormalDiff`, `model_noNormalSkip`, `model_standard`). Our customized network structure in directory `model` performs the best. 
