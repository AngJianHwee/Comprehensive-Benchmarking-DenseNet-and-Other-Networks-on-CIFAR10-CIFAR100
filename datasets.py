from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import logging


model_transforms_dict = {
    'resnet18': [
        transforms.Resize((224, 224)),
    ],
    'densenet': [
        transforms.Resize((224, 224)),
    ],
    'cnn': [
        transforms.Resize((32, 32)),
    ],
    'cnn_batchnorm': [
        transforms.Resize((32, 32)),
    ],
    'cnn_batchnorm_dropout': [
        transforms.Resize((32, 32)),
    ],
    'efficientnet': [
        transforms.Resize((224, 224)),
    ],
    'mobilenetv2': [
        transforms.Resize((224, 224)),
    ],
    'squeezenet': [
        transforms.Resize((224, 224)),
    ],
    'shufflenetv2': [
        transforms.Resize((224, 224)),
    ],
    'vgg': [
        transforms.Resize((224, 224)),
    ],
    'inceptionv3': [
        transforms.Resize((299, 299)),
    ],
    'xception': [
        transforms.Resize((299, 299)),
    ],
    'resnext': [
        transforms.Resize((224, 224)),
    ],
    'alexnet': [
        transforms.Resize((224, 224)),
    ],
    'nasnet': [
        transforms.Resize((331, 331)),
    ],
    'wide_resnet': [
        transforms.Resize((224, 224)),
    ],
}


class RAMDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, classes):
        self.data = data
        self.targets = targets
        self.classes = classes

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


def get_data_loaders(config):
    data_dir = config.get('data_dir', './data')
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 8)
    load_dataset_all_in_ram = bool(
        config.get('load_dataset_all_in_ram', False))

    if 'cifar10' == config['dataset'].lower():
        transform_list = model_transforms_dict.get(config['model'], [])
        if not transform_list:
            raise ValueError(
                f"Model transform not found for model: {config['model']}")
        transform = transforms.Compose(transform_list + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                               download=True, transform=transform)
    elif 'cifar100' == config['dataset'].lower():
        transform_list = model_transforms_dict.get(config['model'], [])
        transform = transforms.Compose(transform_list + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])
        ])

        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                                 download=True, transform=transform)

        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                                download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")

    if load_dataset_all_in_ram:
        # iterate over the dataset and load it all in RAM
        train_data, train_targets = [], []
        logging.info("Loading train set into RAM")
        for data, target in tqdm(trainset, desc="Loading train set into RAM", total=len(trainset)):
            train_data.append(data)
            train_targets.append(target)
        logging.info("Loaded train set into RAM")

        test_data, test_targets = [], []
        logging.info("Loading test set into RAM")
        for data, target in tqdm(testset, desc="Loading test set into RAM", total=len(testset)):
            test_data.append(data)
            test_targets.append(target)
        logging.info("Loaded test set into RAM")

        # now convert the list to tensor and also flatten the targets
        train_data = torch.stack(train_data)
        train_targets = torch.tensor(train_targets)
        test_data = torch.stack(test_data)
        test_targets = torch.tensor(test_targets)

        # create the dataset
        trainset_new = RAMDataset(train_data, train_targets, trainset.classes)
        testset_new = RAMDataset(test_data, test_targets, testset.classes)

        trainloader = torch.utils.data.DataLoader(trainset_new, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(testset_new, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def get_dataset(config):
    # try returning the dataset loader, if it fails, raise error
    try:
        return get_data_loaders(config)
    except KeyError:
        logging.error(f"Unsupported dataset: {config['dataset']}")
        raise ValueError(f"Unsupported dataset: {config['dataset']}")


# def _test_get_data_loaders():
#     # assert if before RAM and after RAM return same thing
#     # use cifar10, get 2 loader, one with RAM and one without
#     config = {
#         'dataset': 'cifar10',
#         'batch_size': 32,
#         'num_workers': 4,
#         'load_dataset_all_in_ram': False
#     }
#     trainloader, testloader = get_data_loaders(config)
#     config['load_dataset_all_in_ram'] = True
#     trainloader_ram, testloader_ram = get_data_loaders(config)

#     # check if the length of both loaders are same
#     assert len(trainloader) == len(trainloader_ram)
#     assert len(testloader) == len(testloader_ram)

#     # iterate 10 times and check if the data is same
#     for i in range(10):
#         data, target = next(iter(testloader))
#         data_ram, target_ram = next(iter(testloader_ram))
#         try:
#             assert torch.all(data == data_ram)
#             assert torch.all(target == target_ram)
#         except AssertionError:
#             print("Data mismatch")
#             # display size
#             print(f"Data: {data.size()}, Data RAM: {data_ram.size()}")
#             # display first 10 elements
#             # print(f"Data: {data[:10]}, Data RAM: {data_ram[:10]}")
#             raise AssertionError

#     print("All tests passed!")


# _test_get_data_loaders()
