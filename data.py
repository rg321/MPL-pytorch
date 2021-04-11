import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from augmentation import RandAugment

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize*0.125),
                              # padding_mode='reflect'
                              ),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(args.data_path, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.train_labels) # targets
    # train_labeled_idxs, train_unlabeled_idxs = x_u_split_test(args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        args.data_path, train_labeled_idxs, train=True,
        transform=transform_labeled
    )

    train_unlabeled_dataset = CIFAR10SSL(
        args.data_path, train_unlabeled_idxs,
        train=True,
        transform=TransformMPL(args, mean=cifar10_mean, std=cifar10_std)
    )

    test_dataset = datasets.CIFAR10(args.data_path, train=False,
                                    transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(args.data_path, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        args.data_path, train_labeled_idxs, train=True,
        transform=transform_labeled
    )

    train_unlabeled_dataset = CIFAR100SSL(
        args.data_path, train_unlabeled_idxs, train=True,
        transform=TransformMPL(args, mean=cifar100_mean, std=cifar100_std)
    )

    test_dataset = datasets.CIFAR100(args.data_path, train=False,
                                     transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all training data
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


def x_u_split_test(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx[label_per_class:])
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])

    np.random.shuffle(labeled_idx)
    np.random.shuffle(unlabeled_idx)
    return labeled_idx, unlabeled_idx


class TransformMPL(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default

        self.ori = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize*0.125),
                                  # padding_mode='reflect'
                                  )])
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize*0.125),
                                  # padding_mode='reflect'
                                  ),
            RandAugment(n=n, m=m)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.train_data[indexs] # data
            self.targets = np.array(self.train_labels)[indexs]

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

# def galaxy_zoo(batch_size=20, test_batch_size=20,
#         dataset_size='normal', resize=400, crop_size=424, network='sqnxt', dataset_type='anp',
#         dataset_source='server_main'):
#     # batch_size=20, test_batch_size=20,
#     #     dataset_size='normal', resize=400, network='sqnxt', dataset_type='anp',
#     #     dataset_source='server_main'
#     from torch.utils.data.sampler import SubsetRandomSampler
#     # batch_size=training_config['batch_size']
#     # test_batch_size=training_config['test_batch_size']
#     # resize=training_config['img_size']
#     # crop_size=training_config['crop_size']
#     transform_train = transforms.Compose([
#             # transforms.Grayscale(num_output_channels=1),
#             transforms.CenterCrop((crop_size,crop_size)),
#             transforms.Resize(resize),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,)),
#         ])

#     gz_root = '/home/nilesh/raghav/' + 'imageFolder'
    
#     gz_dataset = datasets.ImageFolder(root=gz_root
#         ,transform=transform_train)

#     split_1 = .8
#     split_2 = .1
#     shuffle_dataset = True
#     random_seed= 42

#     # Creating data indices for training and validation splits:
#     dataset_size = len(gz_dataset)
#     indices = list(range(dataset_size))
#     split_1 = int(np.floor(split_1 * dataset_size))
#     split_2 = int(np.floor(split_2 * dataset_size))
#     if shuffle_dataset :
#         np.random.seed(random_seed)
#         np.random.shuffle(indices)
#     train_indices, eval_indices, test_indices = indices[:split_1], indices[split_1:(split_1+split_2)], indices[(split_1+split_2):]

#     # Creating PT data samplers and loaders:
#     train_sampler = SubsetRandomSampler(train_indices)
#     eval_sampler = SubsetRandomSampler(eval_indices)
#     test_sampler = SubsetRandomSampler(test_indices)

#     train_loader = DataLoader(gz_dataset
#         ,batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True, sampler=train_sampler)

#     eval_loader = DataLoader(gz_dataset
#         ,batch_size=test_batch_size, shuffle=False, num_workers=1, drop_last=True, sampler=eval_sampler)

#     test_loader = DataLoader(gz_dataset
#         ,batch_size=test_batch_size, shuffle=False, num_workers=1, drop_last=True, sampler=test_sampler)

#     return train_loader, test_loader, gz_dataset

def galaxy_zoo(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize*0.125),
                              # padding_mode='reflect'
                              ),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    gz_root = '/mnt/d/frinks/data-20210410T222155Z-001/data/trainFolder/'
    gz_root1 = '/mnt/d/frinks/data-20210410T222155Z-001/data/testFolder/'
    base_dataset = datasets.ImageFolder(root=gz_root
        # ,transform=transform_train
        )

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, [i[1] for i in base_dataset]) # targets
    # train_labeled_idxs, train_unlabeled_idxs = x_u_split_test(args, base_dataset.targets)

    train_labeled_dataset = galaxy_zooSSL(
        args.data_path, train_labeled_idxs, train=True,
        transform=transform_labeled
    )

    train_unlabeled_dataset = galaxy_zooSSL(
        args.data_path, train_unlabeled_idxs,
        train=True,
        transform=TransformMPL(args, mean=cifar10_mean, std=cifar10_std)
    )

    test_dataset = datasets.ImageFolder(root=gz_root1
        ,transform=transform_val
        )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class galaxy_zooSSL(datasets.ImageFolder):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root=root, 
                            # train=train,
                         transform=transform,
                         target_transform=target_transform,
                         # download=download
                         )
        if indexs is not None:
            self.data = [self[i] for i in indexs]  # data
            self.targets = [self[i][0] for i in indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target




DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'smiling_faces': galaxy_zoo}
