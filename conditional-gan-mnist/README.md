# Conditional Generative Adversarial Networks
## Brief Introduction to Conditional GANs (CGANs)

CGAN is an extension of the standard GAN architecture, where both the generator and discriminator are conditioned on some extra information. In this case, I use mnist digit labels as the conditioning information for generating MNIST-like images based on a given label.

## How Conditional GANs Work ?

### Standard GAN Architecture

A standard GAN consists of two neural networks: `generator` and `discriminator`. These networks are trained simultaneously through a process of adversarial training:

- **Generator**: Takes random noise as input and generates synthetic data.
- **Discriminator**: Takes both real and synthetic data as input and aims to distinguish between the real and synthetic(fake) data.

The generator tries to produce data that is indistinguishable from real data, while the discriminator tries to correctly classify real and synthetic data. This adversarial process continues until the generator produces realistic data that the discriminator can no longer distinguish from real data.

### Conditional GAN Architecture

In a Conditional GAN, one conditions both the generator and the discriminator on additional information, here I conditioned them on class labels. This allows generator to generate data that is not only realistic but also conditioned on specific labels.

Here’s how I incorporate conditioning information into the GAN:

1. **Labels and Images**: Used digit labels from the MNIST dataset as the conditioning information
2. **Label Embeddings**: Convert the labels into embeddings
3. **Concatenation**: Concatenated embeddings with the input noise for the generator and with the real or synthetic images for the discriminator.

![CGAN basic architecture](https://github.com/shoryasethia/GAN/blob/main/conditional-gan-mnist/Conditional-GAN.png)

## Architecture
### Generator 
```
def build_generator():
    
    noise_input_layer = layers.Input(shape=(noise_dim,), name='noise_input_layer')
    noise = layers.Dense(7 * 7 * 128, activation='relu')(noise_input_layer)
    noise = layers.Reshape((7, 7, 128))(noise)
    
    label_input_layer = layers.Input(shape=(1,), name='label_input_layer')
    label_embedding = layers.Embedding(input_dim=10, output_dim=50, name='label_embedding_layer')(label_input_layer)
    label_embedding = layers.Flatten()(label_embedding)
    label_embedding = layers.Dense(7 * 7 * 128, activation='relu')(label_embedding)
    label_embedding = layers.Reshape((7, 7, 128))(label_embedding)

    merge = layers.Concatenate(axis=-1)([noise, label_embedding])

    x = layers.Conv2D(128, activation='relu', padding='same', kernel_size=(3, 3))(merge)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.UpSampling2D()(x)

    x = layers.Conv2D(64, activation='relu', padding='same', kernel_size=(3, 3))(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.UpSampling2D()(x)

    output_layer = layers.Conv2D(1, activation='tanh', padding='same', kernel_size=(3, 3), name='output_layer')(x)

    return models.Model(inputs=[noise_input_layer, label_input_layer], outputs=output_layer, name='generator')
```
### Discriminator
```
def build_discriminator():
  
  input_img_layer = layers.Input(shape=input_shape, name='img_input_layer')
  label_input_layer = layers.Input(shape=(1,), name='label_input_layer')

  label_embedding = layers.Embedding(input_dim=10, output_dim=50, name='label_embedding_layer')(label_input_layer)
  label_embedding = layers.Flatten()(label_embedding)
  label_embedding = layers.Dense(28 * 28, activation='relu')(label_embedding)
  label_embedding = layers.Reshape((28, 28, 1))(label_embedding)

  merge = layers.Concatenate(axis=-1)([input_img_layer, label_embedding])
  
  x = layers.Conv2D(filters = 32,
                    kernel_size=(3,3),
                    strides=(2,2),
                    activation=layers.LeakyReLU(0.2),
                    kernel_initializer = 'he_uniform',
                    padding='same')(merge)
  x = layers.Dropout(0.25)(x)
  
  x = layers.Conv2D(filters = 64,
                    kernel_size=(3,3),
                    strides=(2,2),
                    kernel_initializer = 'he_uniform',
                    padding='same')(x)
  x = layers.BatchNormalization(momentum=0.8)(x)
  x = layers.LeakyReLU(0.2)(x)
  x = layers.Dropout(0.25)(x)
  
  x = layers.Conv2D(filters = 128,
                    kernel_size=(3,3),
                    strides=(2,2),
                    kernel_initializer = 'he_uniform',
                    padding='same')(x)
  x = layers.BatchNormalization(momentum=0.8)(x)
  x = layers.LeakyReLU(0.2)(x)
  x = layers.Dropout(0.25)(x)
  
  x = layers.Conv2D(filters = 256,
                    kernel_size=(3,3),
                    strides=(2,2),
                    kernel_initializer = 'he_uniform',
                    padding='same')(x)
  x = layers.BatchNormalization(momentum=0.8)(x)
  x = layers.LeakyReLU(0.2)(x)
  x = layers.Dropout(0.25)(x)
  
  x = layers.Flatten()(x)
  output_layer = layers.Dense(1, activation = 'sigmoid', name = 'output_layer')(x)
  
  return models.Model(inputs = [input_img_layer, label_input_layer], outputs = output_layer,  name = 'discriminator')
```
## Results of generator over epochs
### Discriminator and Generator Loss curves over epochs
![history vs epochs](https://github.com/shoryasethia/GAN/blob/main/conditional-gan-mnist/history-vs-epoch.png)

### Images generated by generator over epochs on test noise and test labels
![Results over epoch](https://github.com/shoryasethia/GAN/blob/main/conditional-gan-mnist/images-gif.gif)

## Conclusion

My Conditional GAN model effectively generates MNIST-like images based on specified digit labels. This can be particularly useful for generating additional training data or for testing digit recognition models. You can find my implementation in the following repositories:

- [Conditional GAN based trained MNIST model validator](https://github.com/shoryasethia/GAN/tree/main/conditional-gan-mnist)
- Repo for various MNIST's digit recognition models [here](https://github.com/shoryasethia/Digit-Recognition)

### Clone Repo
**Feel free to explore the code and use the trained models for your own projects!**
```
git clone https://github.com/shoryasethia/GAN
```
> * **If you liked anything from this repo, give it a star**
> * **Author : [@shoryasethia](https://github.com/shoryasethia/)**
