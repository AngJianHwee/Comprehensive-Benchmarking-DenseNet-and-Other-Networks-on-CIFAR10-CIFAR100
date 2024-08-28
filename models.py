import ssl
import torchvision.models as models
import pretrainedmodels
from efficientnet_pytorch import EfficientNet as EffNet
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        # freeze the weights of the ResNet layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # replace the last layer with a new fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)

        return x

# densenet


class DenseNet(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet121(pretrained=True)

        # freeze the weights of the DenseNet layers
        for param in self.densenet.parameters():
            param.requires_grad = False

        # replace the last layer with a new fully connected layer
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.densenet(x)

        return x


# a custom 3 layer CNN
class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        # input size is 3x32x32
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64*4*4, 500)
        self.fc2 = nn.Linear(500, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64*4*4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# a custom 3 layer CNN with batch normalization


class CNN_BatchNorm(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_BatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64*4*4, 500)
        self.fc2 = nn.Linear(500, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(500)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64*4*4)
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# a custom 3 layer CNN with batch normalization and dropout


class CNN_BatchNorm_Dropout(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_BatchNorm_Dropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64*4*4, 500)
        self.fc2 = nn.Linear(500, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(500)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64*4*4)
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomEfficientNet, self).__init__()
        self.efficientnet = EffNet.from_pretrained('efficientnet-b0')

        # freeze the weights of the EfficientNet layers
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # replace the last layer with a new fully connected layer
        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.efficientnet(x)

        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV2, self).__init__()
        self.mobilenetv2 = models.mobilenet_v2(pretrained=True)

        # freeze the weights of the MobileNetV2 layers
        for param in self.mobilenetv2.parameters():
            param.requires_grad = False

        # replace the last layer with a new fully connected layer
        in_features = self.mobilenetv2.classifier[1].in_features
        self.mobilenetv2.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.mobilenetv2(x)

        return x

# SqueezeNet


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SqueezeNet, self).__init__()
        self.squeezenet = models.squeezenet1_0(pretrained=True)

        # freeze the weights of the SqueezeNet layers
        for param in self.squeezenet.parameters():
            param.requires_grad = False

        # replace the last layer with a new fully connected layer
        self.squeezenet.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1))

    def forward(self, x):
        x = self.squeezenet(x)

        return x


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(ShuffleNetV2, self).__init__()

        self.shufflenetv2 = models.shufflenet_v2_x1_0(pretrained=True)

        # freeze the weights of the ShuffleNetV2 layers
        for param in self.shufflenetv2.parameters():
            param.requires_grad = False

        # replace the last layer with a new fully connected layer
        in_features = self.shufflenetv2.fc.in_features
        self.shufflenetv2.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.shufflenetv2(x)
        return x


class VGG(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG, self).__init__()
        self.vgg = models.vgg16(pretrained=True)

        # freeze the weights of the VGG layers
        for param in self.vgg.parameters():
            param.requires_grad = False

        # replace the last layer with a new fully connected layer
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.vgg(x)

        return x


class InceptionV3(nn.Module):
    def __init__(self, num_classes=2):
        super(InceptionV3, self).__init__()
        self.inceptionv3 = models.inception_v3(pretrained=True)

        # freeze the weights of the InceptionV3 layers
        for param in self.inceptionv3.parameters():
            param.requires_grad = False

        # replace the last two layers with new fully connected layers
        in_features = self.inceptionv3.fc.in_features
        self.inceptionv3.fc = nn.Linear(in_features, num_classes)

        aux_in_features = self.inceptionv3.AuxLogits.fc.in_features
        self.inceptionv3.AuxLogits.fc = nn.Linear(aux_in_features, num_classes)

    def forward(self, x):
        # Inception v3 outputs are (logit, auxiliary)
        outputs, aux_outputs = self.inceptionv3(x)

        return outputs


class Xception(nn.Module):
    def __init__(self, num_classes=2):
        super(Xception, self).__init__()
        # Note: 'xception' is assumed to be a valid model in pretrainedmodels. Please check and replace it with the correct model name
        self.xception = pretrainedmodels.__dict__['xception'](
            num_classes=1000, pretrained='imagenet')

        # freeze the parameters
        for param in self.xception.parameters():
            param.requires_grad = False

        # replace the last fully-connected layer
        num_ftrs = self.xception.last_linear.in_features
        self.xception.last_linear = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.xception(x)
        return x


class ResNeXt(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNeXt, self).__init__()
        self.resnext = models.resnext50_32x4d(pretrained=True)

        # freeze the weights of the ResNeXt layers
        for param in self.resnext.parameters():
            param.requires_grad = False

        # replace the last layer with a new fully connected layer
        in_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnext(x)

        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)

        # freeze the weights of the AlexNet layers
        for param in self.alexnet.parameters():
            param.requires_grad = False

        # replace the last layer with a new fully connected layer
        in_features = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.alexnet(x)
        return x


class NASNet(nn.Module):
    def __init__(self, num_classes=2):
        super(NASNet, self).__init__()
        self.nasnet = pretrainedmodels.__dict__['nasnetalarge'](
            num_classes=1000, pretrained='imagenet')

        # freeze the weights of the NASNet layers
        for param in self.nasnet.parameters():
            param.requires_grad = False

        # replace the last layer with a new fully connected layer
        in_features = self.nasnet.last_linear.in_features
        self.nasnet.last_linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.nasnet(x)
        return x


class WideResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(WideResNet, self).__init__()
        self.wide_resnet = models.wide_resnet50_2(pretrained=True)

        # freeze the weights of the WideResNet layers
        for param in self.wide_resnet.parameters():
            param.requires_grad = False

        # replace the last layer with a new fully connected layer
        in_features = self.wide_resnet.fc.in_features
        self.wide_resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.wide_resnet(x)

        return x


# dictionary to map model names to model classes
models_dict = {
    'resnet18': ResNet18,
    'densenet': DenseNet,
    
    'cnn': CNN,
    'cnn_batchnorm': CNN_BatchNorm,
    'cnn_batchnorm_dropout': CNN_BatchNorm_Dropout,

    'efficientnet': CustomEfficientNet,
    'mobilenetv2': MobileNetV2,
    'squeezenet': SqueezeNet,
    
    'shufflenetv2': ShuffleNetV2,
    'vgg': VGG,
    'inceptionv3': InceptionV3,

    'xception': Xception,
    'resnext': ResNeXt,
    'alexnet': AlexNet,

    'nasnet': NASNet,
    'wide_resnet': WideResNet,
}


def get_model_methods(config):
    # try return method if not raise error
    try:
        return models_dict[config['model'].lower()]

    except ssl.SSLError:
        ssl._create_default_https_context = ssl._create_unverified_context
        logging.warning("SSL Error: Ignoring SSL certificate verification")
        return models_dict[config['model'].lower()]

    except KeyError:
        logging.error(f"Unsupported model: {config['model']}")
        raise ValueError(f"Unsupported model: {config['model']}")

    except Exception as e:
        logging.error(f"Error: {e}")
        raise ValueError(f"Error: {e}")
