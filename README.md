[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) 

## Notes


```text
        X -----------
        |           |
    weight layer    |
        |           |
    weight layer    |
        |           |
       (+) <---------
        |
       H(X)

Analogy
Blurred Image to Clear Image problem
- Blurred Image + Missing Features = Clear Image
- Missing Features = Clear Image - Blurred Image 

Solve for Missing Features
F(X) = H(X) - X
```


When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly. 
Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error, as reported in [11, 42] and thoroughly verified by our experiments.

In this paper, we address the degradation problem by introducing a deep residual learning framework. Instead of hoping each few stacked layers directly fit a desired underlying mapping, we explicitly let these layers fit a residual mapping. 

Formally, denoting the desired underlying mapping as H(x), we let the stacked nonlinear layers fit another mapping of F(x) := H(x)−x. The original mapping is recast into F(x)+x. 

We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping. To the extreme, if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers.

## Goals?
- Our extremely deep residual nets are easy to optimize, but the counterpart “plain” nets (that simply stack layers) exhibit higher training error when the depth increases; 
- Our deep residual nets can easily enjoy
accuracy gains from greatly increased depth, producing results substantially better than previous networks


The identity shortcuts can be directly used when the input and output are of the same dimensions. 
When the dimensions increase, we consider two options: 
- (A) The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions. This option introduces no extra parameter; 
- (B) The projection shortcut in Eqn.(2) is used to match dimensions (done by 1×1 convolutions). 

For both options, when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2.

(A) zeropadding shortcuts are used for increasing dimensions, and all shortcuts are parameterfree (the same as Table 2 and Fig. 4 right); 
(B) projection shortcuts are used for increasing dimensions, and other shortcuts are identity;
(C) all shortcuts are projections.  Table 3 shows that all three options are considerably better than the plain counterpart. B is slightly better than A. We argue that this is because the zeropadded dimensions in A indeed have no residual learning. C is marginally better than B, and we attribute this to the extra parameters introduced by many (thirteen) projection shortcuts. 

But the small differences among A/B/C indicate that projection shortcuts are not essential for addressing the degradation problem. So we do not use option C in the rest of this paper, to reduce memory/time complexity and model sizes. 

## BottleNeck for Resnet-50 / ImageNet

- Deeper Bottleneck Architectures. 
Next we describe our deeper nets for ImageNet. Because of concerns on the training time that we can afford, we modify the building block as a bottleneck design4.

For each residual function F, we use a stack of 3 layers instead of 2.
The three layers are 1×1, 3×3, and 1×1 convolutions, where the 1×1 layers are responsible for reducing and then increasing (restoring) dimensions, leaving the 3×3 layer a bottleneck with smaller input/output dimensions.

Use option (B) to increase dimensions. 

Resnet34 (2 residual blocks) to = 34 layer net with this 3 layer bottleneck blocks (Resnet50)

## Dataset
CIFAR-10 dataset, which consists of 50k training images and 10k testing images in 10 classes


In this paper, we use no maxout/dropout and just simply impose regularization via deep and thin architectures by design, without distracting from the focus on the difficulties of optimization.


## Expected Output

Plain vs Residual Error vs Epoch Graph (Both options A (extra padding) and B (no extra padding just match it (inp) with relevant layers of output))

B > A since zero padded dimensions in A have no residual learning.
 
Increasing Depth
Plain20,32,44,56 Layer nets vs Residual20,32,44,56 Nets

The residual networks use Option A, which means they have exactly the same number of trainable parameters as their plain counterparts.

More on these options / shortcuts to match the dimensions:

- Option A: Zero-padding

Upon downsampling, the number of feature maps doubles and the side length of each feature map is halved. Pad the original input's channels by concatenating extra zero-valued feature maps. Match the new, smaller feature map size by pooling using a 1x1 kernel with stride 2.

- Option B: Linear Projections

Use a convolutional layer with 1x1 kernels and stride 2 to linearly project the N input channels to 2N output channels. Abstracting each feature map as a single element, the linear projection can be thought of as a 2D operation:

- Option C: More Linear Projections
Use the linear projections described in Option B for every shortcut, not just those that down sample. This introduces more trainable parameters, which [1] argues to be the reason that Option C marginally outperforms Option B

Graphs:

[Graph](graph.png)