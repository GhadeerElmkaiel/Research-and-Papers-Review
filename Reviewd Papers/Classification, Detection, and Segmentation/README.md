## A survey of loss functions for semantic
In this paper different loss functions for segmentation were presented (15 different loss function). Each loss function was described, and its best used cases were showed.
There are multiple useful loss functions for different cases (Biased datasets, shape matching, etc.).
________________________________________________________________
## Where is My Mirror
This paper presented (**what was in 2019**) a novel method for segmenting mirrors from *Images* using **MirrorNet** neural network.
The main Idea of MirrorNet is to utilize  multilevel features (low level, high level) and merge them to get the final result.
The main structure of the network is as follows:
- Feature extraction backbone (in this case was **ResNet-101**).
- Multilevel CCFE modules (each for a certain level of feature gotten from the feature extractor)
- CCFE modules "*Contextual Contrasted Features Extraction*" are followed by **Deconve** , **Attention** , **Conv**, then **Sigmoid** to get the result mask.
- Each mask (starting from the highest level (*smallest dimensions*)) is an additional input for the next level CCFE module (*it acts as a mask for the input of the next level*)
### CCFE modules:
Each module consists of four chained CCFE blocks, and the output of each CCFE block are fused via an attention module to generate multi-level contextual contrasted features. *Contrasted to find the inconsistence to detect the mirrors.*

### Problems and Disadvantages:
- In each CCFE module there are 4 CCFE blocks, in each block the model uses **Four BatchNorm2d modules** which dramatically increase the variance of the training process (because it is batch dependent). Especially when the batch size is very small as in the case of huge neural networks as MirrorNet or GDNet.
________________________________________________________________
## Progressive Mirror Detection
The main contributions of this paper are:
- Relation Contextual Contrasted Local modules (*RRCL modules*)
- Edge Detection and Fusion modules. (*EDF modules*)
The *RRCL modules* is different than  CCFE, because it does not only considers the contrast, but the similarities too.
They consider the **Global Similarity (Global Relation)** of the global features extracted using *Global Feature Extractor* **GFE**, and they consider the **Local Contrast** using local features extracted using *Local Feature Extractor* **LFE**.
### RRCL modules:
It consists of **Global Relation** and **Contextual Contrast Local ** which are multiplied to find the possible mirror regions.
The **Contextual Contrast Local ** is gotten by subtracting the *Local features* from the *Contextual features* (The differ in the dilation values for different levels)

### EDF modules:
It is designed to extract multi-scale mirror edge features.
It uses *Low Level* and *High Level* features to produce a boundary map.

