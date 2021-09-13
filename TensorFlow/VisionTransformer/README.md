# Vision Transformers with TensorFlow-DirectML


## What are Vision Transformers? 

Vision Transformers (ViT) were introduced in the paper An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. It was the first work to have achieved excellent results on training a Transformer Encoder on ImageNet for Image Classification: Vision Transformers models pre-trained on the JFT300M dataset outperformed state of the art ResNet baselines like BiT while using significantly lower computational resources to transfer learn. Since then there have been a number of advances in the vision transformers domain, such as Swin Transformers achieving state of the art results in Object Detection and Semantic Segmentation, among other vision tasks.  
 

## Core idea behind a Vision Transformer:  
The core idea is surprisingly simple: we split an image into smaller equal-sized patches and feed these in as a sequence to stacked Transformer Encoder Blocks. The context vector for each patch after self-Attention is a weighted sum over the other context vectors and the weights, are learned to capture relationships between patches. 
 

## Step 1: Extracting Patches 
With a pre-defined patch size and stride, we move over the image extracting the (PATCH_SIZE x PATCH_SIZE x channels) dimensional patches. In our case, each patch has the shape (4, 4, 3). Next, we flatten the patch into a single 48-dimensional vector as input to our patch_projection layer.  


## Step 2: Patch Embeddings 
The patch projection Dense layer (self.patch_proj) embeds each original flattened patch into an (embedding_dim)-sized embedding.  


## Step 3: Positional Encodings 
In transformers, words (or in this case, patches) are processed in parallel instead of maintaining a sequential hidden state as is the case in a RNN or LSTM. Imagine randomly permuting the image patches and then passing them into the model. This permuted image may or may not be semantically similar to the original image. Clearly, we need a way to encode the relative positions in which the patches appear in the image. Thus, we define a learnable Embedding layer that encodes positions to embeddings of shape embedding_dim. Adding this positional encoding enables the patch embeddings to also represent positional information. This is similar to how positional encodings that represent the index of the word in the sentence are added to word embeddings before being fed into a Transformer model like BERT. 


## Step 4: Encoder: Multi-head Attention Transformer Blocks 
The patch embeddings are fed into a series of Transfomer Encoder Blocks, each of which has NUM_HEADS heads. Each Transformer Block contains MultiHeadSelfAttention Layer which takes in a sequence of patch embedding vectors and returns a sequence of context vectors of the same length. After going through a feed forward network (MLP), this sequence of context vectors is fed to the next Transformer Encoder Block, and so on. After being passed to the final Transformer Block, the context vectors are flattened and then passed through a MLP classifier that outputs a 10 or 100 dimensional prediction vector. 

 

## Step 5: Loss and Gradient Descent 
Since we have a multi-class classification task, the loss function we’ll use is SparseCategoricalCrossEntropy loss. For our gradient descent updates, we’ll use the Adam optimizer. Adam (Adaptive Moment Estimation) is a gradient decent optimization algorithm that computes adaptive learning rates for each parameter. Adam stores and leverages both an exponentially decaying average of past squared gradients vt well as an exponentially decaying average of past gradients mt , which plays the role of momentum. Check out the references section for more details on the Adam optimizer. 

### Note: GELU 
In the feed-forward network within each Transformer Block, we utilize the GELU activation function. GELU stands for Gaussian Error Linear Unit and GELU(x) is defined as x*CDF(x) where x~ N(0,1). GELU is a commonly used activation function used in Transformer models like BERT and GPT-3. 


## Data Pipeline 
In order to build an optimized training data pipeline, we leverage TF Dataset. TensorFlow allows us to define operations like cache() and prefetch() to help us significantly reduce the CPU batch loading wait-time. Furthermore, we’ll leverage the map() operation to randomly apply data augmentations to create an augmented version of our training dataset. 


## Data Augmentation  
In order to prevent overfitting on the training set as well as to generate additional samples, we will leverage a number of image data augmentation techniques in data_augmentations.py. These include random_rotation, random_contrast, random_hue, random_shear and others. You should try experimenting with these data augmentation techniques to further improve your validation and test accuracy. Additional details about each augmentation can be found in the code comments. 

 

# Training 

## Datasets: CIFAR-10 and CIFAR-100 

The CIFAR-10 dataset consists of 60000 32x32 colour images across 10 mutually-exclusive classes such as airplane, automobile, cat, bird, etc. Each class as 5000 training images and 1000 test images. 
The CIFAR-100 dataset as 60000 32x32 images across 100 classes, each class containing 600 images. Each class has 500 training images and 100 test images. Within the train_cifar10.py and train_cifar100.py, CIFAR-10 and CIFAR-100 are downloaded using tf.keras.datasets.


## Note: Device Placement 
The environment variable DML_VISIBLE_DEVICES controls the GPU device that you want your model to train on. If you want to switch to another graphics card, you can set the environment variable accordingly to "1", "2", etc. Check out the FAQ @ https://docs.microsoft.com/en-us/windows/ai/directml/gpu-faq for more information.

    import os 
    os.environ["DML_VISIBLE_DEVICES"] = "0" 


### Train and Evaluate

I) To Train and evaluate on cifar10 

    python train_cifar10.py 

II) To Train and evaluate on cifar100 

    python train_cifar100.py 

The default hyperparameters are: 
    
    PATCH_SIZE= 4
    PATCH_STRIDE=4
    NUMBER_OF_LAYERS=8
    EMBEDDING_DIM=64
    NUM_HEADS= 8
    MLP_HIDDEN_DIM= 256
    LEARNING_RATE= 0.001
    BATCH_SIZE= 512
    EPOCHS= 100
    PATIENCE= 10

You can change these within the train_cifar*.py files. 

## Conclusion 

As you have probably noticed, the test accuracy we obtained is not close to the current state of the art on Cifar-10. To achieve the state of the art Image Classification results reported in the “An Image is 16 x 16 words” paper, we require pre-training the model on the JFT-300M proprietary dataset and then fine-tuning on CIFAR-10 and CIFAR-100. Vision Transformers perform best when pre-trained on a huge, higher-resolution dataset before being fine-tuned on the target dataset. That being said, there are a number of settings you can play with in this augmentation such as different patch sizes and strides, a different learning rate schedule, trying different embedding and layer dimensions within the Transformer Block, and applying additional data augmentations.  


## References: 

Portions of this sample were sourced from (I) and (II). The original sources are Copyright (c) The TensorFlow Authors and Copyright (c) Khalid Salama, see NOTICE for more information.

(I) TensorFlow Transformer Tutorial (Apache License 2.0)
https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb 

(II) Keras Vision Transformers Example (Apache License 2.0)
https://github.com/keras-team/keras-io/tree/master/examples



## Resources (to learn more!): 

(I) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale 
https://arxiv.org/abs/2010.11929 

(II) Attention is All You Need 
https://arxiv.org/abs/1706.03762 

(III)The Illustrated Transformer 
https://jalammar.github.io/illustrated-transformer/ 

(IV) Gradient descent optimization algorithms such as Adam, AdaMax, and more RMSprop 
https://ruder.io/optimizing-gradient-descent/ 

(V) CIFAR-10 and CIFAR-100 
https://www.cs.toronto.edu/~kriz/cifar.html 

(VI) Optimizing data loading pipeline using tf.data.Dataset 
https://www.tensorflow.org/guide/data_performance 