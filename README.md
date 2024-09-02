# Comprehensive Benchmarking: DenseNet and Other Networks on CIFAR10 & CIFAR100

This project include Benchmarking study of 13 well-known ML models including ResNet, Densenet, Custom CNN (with and without batch norm and dropout), EfficientNet, and others on CIFAR10 and CIFAR100 datasets.

This project provides a detailed examination of each model's capability in classifying image data, offering insights into their distinct strengths and weaknesses.

# Supported Models

Currently, the following models are supported in this project:

- ResNet18
- DenseNet
- Custom CNN
- Custom CNN with Batch Normalization
- Custom CNN with Batch Normalization and Dropout
- EfficientNet
- MobileNetV2
- SqueezeNet
- ShuffleNetV2
- VGG
- InceptionV3
- ResNeXt
- AlexNet
- NASNet
- WideResNet


# Steps on implementing your own model

First implement the model and register it in `models.py` file and update the `models_dict` dictionary.

Next, update the `model_transforms_dict` dictionary in `datasets.py` file with the appropriate transformations for your model.

