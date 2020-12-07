# The Neural SLAM model.
The neural network of the Neural SLAM consist of **Feature extraction part**, **Map prediction part**, and **Position estimator**.
then there  and the second for Position estimation)
## Feature extraction part
the feature extraction part is a convolution part then linear layers that extracts features:
1- first 8 layers of resnet18 (pretrained) output shape will be **(7x7)** with **512** channels
2- conv2d -> output **64** channels
3- ReLU layer
4- Flatten Layer
5- Linear layer ->  output **1024**
6- ReLU layer
7- Dropout(0.5) layer
8- Linear layer ->  output **4096**
9- ReLU layer

## Map prediction deconvolutional network
the map prediction is a deconvolution part to create a map prediction using the features from the previous part. this network consist of the following parts:
0- Change the shape from [4096] to [-1 (**batchsize**), 64, 8, 8]
1- ConvTranspose2d (out_channels: **32**, kernel_size: **(4x4)**, stride **(2, 2)**, padding: **(1, 1)**)
2- ReLU layer
3- ConvTranspose2d (out_channels: **16**, kernel_size: **(4x4)**, stride **(2, 2)**, padding: **(1, 1)**)
4- ReLU layer
5- ConvTranspose2d (out_channels: **2**, kernel_size: **(4x4)**, stride **(2, 2)**, padding: **(1, 1)**)
6- Torch Sigmoid layer

The final two layers are: (the first is for predicted map (**free space**), and the second is for the explored area (**free space and obstacles**)

## Pose Estimator convolutional network
This part predict the position of the robot using the current local map and the previous local map after transform it to match the current local map and it consist of 
1- ConvTranspose2d (out_channels: **64**, kernel_size: **(4x4)**, stride **(2, 2)**) *The input for the this layer is the current local predicted map (two layers), and the last local predicted map (two layers)*
2- ReLU layer
3- ConvTranspose2d (out_channels: **32**, kernel_size: **(4x4)**, stride **(2, 2)**)
4- ReLU layer
5- ConvTranspose2d (out_channels: **16**, kernel_size: **(4x4)**, stride **(1, 1)**)
6- Flatten layer
7- Linear layer output **1024**
8- Three Linear layers (one for x, one for y, and one for the rotation) (**1024 -> 128**)
8- Three Linear layers (one for x, one for y, and one for the rotation) (**128 -> 1**)