# Description: This file contains the code to attack a real dataset using the method described in the paper.
import os

# import sys
import hydra
import time
import logging
import copy

from logging import INFO  # , DEBUG
from omegaconf import DictConfig, OmegaConf, ListConfig

import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms
from torch.utils.data import Subset

from solving_hssp import main
from solving_hssp.src import hidden_lattice

from sage.all import matrix, Integers, ZZ, Integer, randint

# ############################ Constants ############################ #
ID_TO_LABELS = {
    1: "Boston Housing",
    60: "Liver Disorders",
    477: "Real Estate Valuation",
    555: "Apartment for Rent Classified",  # Removed: complicated dataset
    597: "Productivity Prediction of Garment Employees",
    "MNIST": "MNIST",
    "BostonHousing": "Boston Housing",
    "UTKFace": "UTKFace",
    "SalaryDataset": "Salary Dataset",
}

ID_TO_PATH = {
    1: "~/data/BostonHousing",
    60: "~/data/LiverDisorders",
    477: "~/data/RealEstateValuation",
    555: "~/data/ApartmentforRentClassified",  # Removed: complicated dataset
    597: "~/data/ProductivityPredictionofGarmentEmployees",
    "MNIST": "~/data",
    "BostonHousing": "~/data/BostonHousing",
    "UTKFace": "~/data/UTKFace",
    "SalaryDataset": "~/data",
}


# ############################ Utils ############################ #
def my_subdir_suffix_impl(
    task_overrides: ListConfig,
    exclude_patterns: ListConfig,
) -> str:
    """
    Filter out the task_overrides with the exclude_patterns and return the remaining overrides as a string.
    It is meant to be used as a resolver in OmegaConf for Hydra.

    Source Code
    ---------------
        https://github.com/facebookresearch/hydra/issues/1873

    Parameters
    ----------
    task_overrides : ListConfig
        List of overrides to process.
    exclude_patterns : ListConfig
        List of patterns to exclude.

    Examples
    --------
        >>> from omegaconf import OmegaConf
        >>> task_overrides = OmegaConf.create(["a=1", "b=2", "c=3"])
        >>> exclude_patterns = OmegaConf.create(["b"])
        >>> my_subdir_suffix_impl(task_overrides, exclude_patterns)
        'a=1_c=3'
    """
    import re

    rets: list[str] = []
    for override in task_overrides:
        should_exclude = any(re.search(exc_pat, override) for exc_pat in exclude_patterns)
        if not should_exclude:
            rets.append(override)

    return "_".join(rets)


def drop_na(X, y):
    X_idx_na = X.isna()
    X_idx_na = np.where(X_idx_na.any(axis=1))[0]
    X = X.drop(index=X_idx_na)
    y = y.drop(index=X_idx_na)

    y_idx_na = y.isna()
    y_idx_na = np.where(y_idx_na)[0]
    X = X.drop(index=y_idx_na)
    y = y.drop(index=y_idx_na)
    return X, y


def replace_function(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


def vizualize_target(y, labels):
    nb_rows_columns = math.ceil(math.sqrt(len(labels)))
    fig, axes = plt.subplots(nb_rows_columns, nb_rows_columns, figsize=(15, 10))
    axes = axes.flatten()

    for i, targets in enumerate(y):
        np_targets = np.array(targets)
        # hist, bins = np.histogram(np_targets, bins=10)
        axes[i].hist(np_targets, bins=100, alpha=0.5)
        axes[i].set_title(labels[i])
        axes[i].set_xlabel("Target")
    plt.show()


def almost_unique(tensor, treshold=1e-2):
    tensor = tensor.sort().values
    unique_tensor = []
    for i in range(len(tensor)):
        if i == 0 or abs(tensor[i] - tensor[i - 1]) > treshold:
            unique_tensor.append(tensor[i])
    return torch.tensor(unique_tensor)


def compute_batch_gradients(model, batch, loss_fn):
    model.zero_grad()
    x, y = batch
    z_0, z_1, z_2 = model(x)
    loss = loss_fn(z_0, y)
    # logging.log(INFO, f"{z_0=}, {y=}, {loss=}")
    e = (z_0 - y).t()
    gradient = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
    return gradient, z_0, z_1, z_2, e, loss


def compute_P(db_0, db_1, w_0, batch_size):
    return batch_size / 2 * torch.concat([db_0, db_1 / w_0.reshape(-1)], dim=0)


def compute_density(dataset, K):
    digits_of_dataset = [
        len(str(dataset.iloc[i]).replace(".", "").lstrip("0").rstrip("0")) for i in range(len(dataset))
    ]
    mean_digits = np.mean(digits_of_dataset)
    max_digits = np.max(digits_of_dataset)
    return {"mean": round(K / math.log(10**mean_digits, 2), 4), "max": round(K / math.log(10**max_digits, 2), 4)}


# ############################ Dataset and Preprocessing ############################ #


def load_mnist():
    MNIST_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    MNIST_TARGET_TRANSFORM = transforms.Compose([transforms.Lambda(lambda x: torch.tensor([x]))])
    dataset = MNIST(
        ID_TO_PATH["MNIST"],
        train=True,
        transform=MNIST_TRANSFORM,
        target_transform=MNIST_TARGET_TRANSFORM,
        download=False,
    )
    return dataset


def load_dataset(id):
    if id == "SalaryDataset":
        dataset = copy.deepcopy(pd.read_csv(ID_TO_PATH[id] + "/Salary_Data.csv"))
        dataset.dropna(inplace=True)
        X = dataset.drop(columns=["Salary"], inplace=False)
        y = dataset["Salary"]
    else:
        X = copy.deepcopy(pd.read_csv(ID_TO_PATH[id] + "/inputs.csv"))
        y = copy.deepcopy(pd.read_csv(ID_TO_PATH[id] + "/targets.csv"))
        X, y = drop_na(X, y)
    return X, y


def preprocess_dataset(df, normalize="classic"):
    df.reset_index(drop=True, inplace=True)

    # Categorial columns
    columns_to_one_hot = [
        col for col in df.columns if not isinstance(df[col].iloc[0], (int, float, np.float64, np.int64))
    ]
    new_df = pd.get_dummies(df, columns=columns_to_one_hot)

    # Numerical columns
    columns_to_normalize = [
        col for col in df.columns if isinstance(df[col].iloc[0], (int, float, np.float64, np.int64))
    ]

    if normalize == "0-1":
        new_df[columns_to_normalize] = new_df[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    if normalize == "classic":
        new_df[columns_to_normalize] = new_df[columns_to_normalize].apply(
            lambda x: (x - x.mean()) / x.std()
        )  # more effective
    return new_df


class MNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNIST, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = super(MNIST, self).__getitem__(index)
        return index, img, target


class UTKDataset(Dataset):
    """
    Inputs:
        dataFrame : Pandas dataFrame
        transform : The transform to apply to the dataset
    """

    def __init__(self, path_to_data, setup={}):

        dataFrame = pd.read_csv(path_to_data + "/age_gender.csv")
        # read in the transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.49,), (0.23,))])
        self.transform = transform

        # Use the dataFrame to get the pixel values
        data_holder = dataFrame.pixels.apply(lambda x: np.array(x.split(" "), dtype=float))
        arr = np.stack(data_holder)
        arr = arr / 255.0
        arr = arr.astype("float32")
        arr = arr.reshape(arr.shape[0], 48, 48, 1)
        # reshape into 48x48x1
        self.data = arr

        # get the age, gender, and ethnicity label arrays
        self.age_label = np.array(dataFrame.age[:])  # Note : Changed dataFrame.age to dataFrame.bins
        self.gender_label = np.array(dataFrame.gender[:])
        self.eth_label = np.array(dataFrame.ethnicity[:])

        self.setup = setup

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # load the data at index and apply transform
        data = self.data[index]
        data = self.transform(data)

        # load the labels into a list and convert to tensors
        # labels = torch.tensor((self.age_label[index], self.gender_label[index], self.eth_label[index]))
        labels = torch.tensor([self.age_label[index]])  # WARNING: Changed to only age

        # return data labels
        return (torch.tensor(index), data.to(**self.setup), labels.to(**self.setup))


class RegressionDataset(Dataset):
    def __init__(
        self,
        df_input,
        df_target,
        transform=None,
        target_transform=None,
        setup={"dtype": torch.float64, "device": "cpu"},
    ):
        self.inputs = np.array(df_input).astype(float)  # WARNING
        self.targets = np.array(df_target).astype(float).reshape(-1, 1)  # WARNING

        if transform is None:
            self.transform = lambda x: torch.tensor(x)
        else:
            self.transform = transform

        if target_transform is None:
            self.target_transform = lambda x: torch.tensor(x)
        else:
            self.target_transform = target_transform

        self.setup = setup

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(idx),
            self.transform(self.inputs[idx]).to(**self.setup),
            self.target_transform(self.targets[idx]).to(**self.setup),
        )


class SalaryDataset(Dataset):
    """Salary Dataset"""

    diplomas_map = {
        "High School": 0,
        "Bachelor's": 1,
        "Bachelor's Degree": 1,
        "Master's": 2,
        "Master's Degree": 2,
        "PhD": 3,
        "phD": 3,
    }

    def __init__(self, data_path, is_one_hot=True, transform=None, target_transform=None, setup={}):
        self.data_path = data_path
        self.dataset = copy.deepcopy(pd.read_csv(f"{data_path}/Salary_Data.csv"))
        self.dataset.dropna(inplace=True)
        self.dataset = preprocess_dataset(self.dataset, normalize="classic")
        self.dataset = self.dataset.astype(float)

        logging.log(INFO, f"Dataset: {self.dataset.head()}")

        # Education Level
        # self.dataset["diploma"] = self.dataset["Education Level"].map(self.diplomas_map)
        # self.dataset.drop(columns=["Education Level"], inplace=True)

        # # Gender
        # genders = self.dataset["Gender"].unique()
        # self.genders_map = {gender: i for i, gender in enumerate(genders)}
        # self.dataset["Gender"] = self.dataset["Gender"].map(self.genders_map)

        # # Job Title
        # job_titles = self.dataset["Job Title"].unique()
        # self.job_titles_map = {job: i for i, job in enumerate(job_titles)}
        # self.dataset["Job Title"] = self.dataset["Job Title"].map(self.job_titles_map)

        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

        if target_transform is None:
            # Mean = 115326.96
            # Std = 52786.18
            # self.target_transform = transforms.Lambda(lambda y: (y - Mean) / (Std))
            self.target_transform = lambda y: y
        else:
            self.target_transform = target_transform

        self.setup = setup

        # self.inputs = torch.from_numpy(self.dataset.drop(columns=["Salary"], inplace=False).values).to(**self.setup)
        # self.targets = torch.from_numpy(self.dataset["Salary"].values).to(**self.setup)

        # # num_classes = [int(max(self.dataset["Age"])+1)]
        # num_classes = [int(max(self.inputs[:, i]) + 1) for i in range(len(self.inputs[0]))]
        # self.num_classes = [0] + num_classes
        # self.is_one_hot = is_one_hot

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.is_one_hot:
            zz = torch.zeros((sum(self.num_classes),), **self.setup)
            for i in range(len(self.inputs[idx])):
                begin = sum(self.num_classes[: i + 1])
                end = self.num_classes[i + 1]
                one_hot = F.one_hot(self.inputs[idx][i].long(), self.num_classes[i + 1])
                zz[begin : begin + end] = one_hot
            return (torch.tensor(idx), self.transform(zz), self.target_transform(self.targets[idx]))
        else:
            return (torch.tensor(idx), self.transform(self.inputs[idx]), self.target_transform(self.targets[idx]))


def get_dataloader(id, batch_size, max_size=None, shuffle=True, setup={}):
    if isinstance(id, int) or id == "BostonHousing" or id == "SalaryDataset":
        df_input, df_target = load_dataset(id)
        df_input = preprocess_dataset(df_input)
        if max_size is None:
            dataset = RegressionDataset(df_input, df_target, setup=setup)
        else:
            dataset = RegressionDataset(df_input[:max_size], df_target[:max_size], setup=setup)
        trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        input_shape = dataset[0][1].shape[0]
    elif id == "MNIST":
        dataset = load_mnist()
        if max_size is None:
            trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        else:
            trainloader = DataLoader(
                Subset(dataset, random.sample(range(len(dataset)), max_size)),
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,
            )
        input_shape = 28 * 28
    elif id == "UTKFace":
        dataset = UTKDataset(ID_TO_PATH["UTKFace"], setup=setup)
        if max_size is None:
            trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        else:
            trainloader = DataLoader(
                Subset(dataset, random.sample(range(len(dataset)), max_size)),
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,
            )
        input_shape = 48 * 48
    # elif id == "SalaryDataset":
    #     dataset = SalaryDataset(ID_TO_PATH["SalaryDataset"], setup=setup)
    #     if max_size is None:
    #         trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    #     else:
    #         trainloader = DataLoader(
    #             Subset(dataset, random.sample(range(len(dataset)), max_size)),
    #             batch_size=batch_size,
    #             shuffle=shuffle,
    #             drop_last=True,
    #         )
    #     input_shape = dataset[0][1].numel()
    return trainloader, input_shape


def mean_std_estimated_z_0(model, trainloader, setup):
    estimated_z_0 = []
    for batch in trainloader:
        idx, data, target = batch
        data, target = data.to(**setup), target.to(**setup)
        z_0, z_1, z_2 = model(data)
        estimated_z_0.append(z_0)
        # sys.exit(0)

    estimated_z_0 = torch.cat(estimated_z_0, dim=0)
    mean_estimated_z_0 = torch.mean(estimated_z_0)
    std_estimated_z_0 = torch.std(estimated_z_0)
    return mean_estimated_z_0, std_estimated_z_0


def get_all_vectors(trainloader, model, batch_size, testing=False, setup={}):
    # # P = delta * e # #
    all_P = []
    all_delta = []
    all_e = []
    all_idx = []
    all_target = []
    all_z_0 = []
    all_goal = []

    w_0 = model.fc0.weight
    loss_fn = nn.MSELoss()

    total = 0
    for count, (idx, input, target) in enumerate(trainloader):
        if count % max(len(trainloader) // 10, 1) == 0:
            logging.log(INFO, f"Computing batch gradients for batch {count}/{len(trainloader)}")
        # Model Input
        input = input.to(**setup)
        target = target.to(**setup)

        # Forward pass
        grads, z_0, z_1, _, e, _ = compute_batch_gradients(model, (input, target), loss_fn)
        _, _, _, db_1, _, db_0 = grads[-6:]

        # Compute the Client's Private Information
        delta = (z_1 > 0).to(**setup).t()
        e = (z_0 - target).t()

        # Compute Attackers' Information : P and Goal
        P = compute_P(db_0, db_1, w_0, batch_size)
        goal = P.abs().max().item()

        # Process
        all_e.append(e.detach())
        all_delta.append(delta.unsqueeze(0).detach())
        all_P.append(P.unsqueeze(0).detach())
        all_goal.append(torch.tensor([goal], **setup))
        all_idx.append(idx.unsqueeze(0).detach())
        all_z_0.append(z_0.detach())
        all_target.append(target.detach())
        total += batch_size

        if testing:
            break

    return {
        "all_idx": all_idx,
        "all_e": all_e,
        "all_delta": all_delta,
        "all_P": all_P,
        "all_goal": all_goal,
        "total": total,
        "all_target": all_target,
        "all_z_0": all_z_0,
    }


# ############################ Model ############################ #


def get_model(input_shape, n, m, hidden_layer=0, model_name=None):
    if model_name is None or model_name == "MLP":
        return MLP(input_shape, n, m, hidden_layer)
    elif model_name.lower() in model_lookup_table:
        return model_lookup_table[model_name.lower()](n, m)
    else:
        raise ValueError(f"Model {model_name} not found in the lookup table.")


class MLP(nn.Module):
    def __init__(self, input_shape, n, m, hidden_layers=0):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.fc2 = nn.Linear(input_shape, n)
        self.fchidden = nn.Sequential(*[nn.Sequential(nn.Linear(n, n), nn.ReLU()) for _ in range(hidden_layers)])
        self.fc1 = nn.Linear(n, m)
        self.fc0 = nn.Linear(m, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        z_3 = F.relu(self.fc2(x))
        z_2 = self.fchidden(z_3)
        z_1 = F.relu(self.fc1(z_2))
        z_0 = self.fc0(z_1)
        return z_0, z_1, z_2


class McMahan_CNN(nn.Module):
    """Convolutional Neural Network architecture as described in McMahan 2017
    paper :
    [Communication-Efficient Learning of Deep Networks from
    Decentralized Data] (https://arxiv.org/pdf/1602.05629.pdf)

    Expected input_size: [N,1,28,28]
    """

    def __init__(self, n, m) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, n)
        self.fc2 = nn.Linear(n, m)
        self.fc0 = nn.Linear(m, 1)  # Regression task

        self.input_shape = torch.Size([1, 28, 28])
        self.output_shape = torch.Size([1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """expect input of size [N, 1/3, 28, 28]."""
        output_tensor = F.relu(self.conv1(input_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = F.relu(self.conv2(output_tensor))
        output_tensor = self.pool(output_tensor)
        z_2 = nn.Flatten()(output_tensor)
        z_2 = F.relu(self.fc1(z_2))
        z_1 = F.relu(self.fc2(z_2))
        z_0 = self.fc0(z_1)
        return z_0, z_1, z_2


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = torch.sigmoid(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.sigmoid(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = torch.sigmoid(self.bn1(self.conv1(x)))
        out = torch.sigmoid(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.sigmoid(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, n, m, is_gray=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        if is_gray:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc2 = nn.Linear(512 * block.expansion, n)
        self.fc1 = nn.Linear(n, m)  # WARNING: Added Layer for the attack
        self.fc0 = nn.Linear(m, 1)  # WARNING: Regression task
        self.output_shape = torch.Size([1])

        self.init_params()

    def init_params(self):
        """Initialize the parameters in the ResNet model"""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode="fan_out", nonlinearity="relu")
                # if m.bias is not None:
                #     nn.init.constant(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # if m.bias is not None:
                #     nn.init.constant(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.sigmoid(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        z_3 = out.view(out.size(0), -1)
        z_2 = F.relu(self.fc2(z_3))
        z_1 = F.relu(self.fc1(z_2))
        z_0 = self.fc0(z_1)
        return z_0, z_1, z_2


class ResNet18(ResNet):
    """
    torchsummary.summary(ResNet18(num_classes=10), input_size=(3,28,28))
    ================================================================
    Total params: 11,173,962
    Trainable params: 11,173,962
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 8.79
    Params size (MB): 42.63
    Estimated Total Size (MB): 51.42
    ----------------------------------------------------------------
    """

    def __init__(self, n, m, is_gray=True):
        super().__init__(BasicBlock, [2, 2, 2, 2], n, m, is_gray)
        if is_gray:
            self.input_shape = torch.Size([1, 28, 28])
        else:
            self.input_shape = torch.Size([3, 28, 28])


class ResNet34(ResNet):
    def __init__(self, n, m, is_gray=True):
        super().__init__(BasicBlock, [3, 4, 6, 3], n, m, is_gray)


class ResNet50(ResNet):
    def __init__(self, n, m, is_gray=True):
        super().__init__(Bottleneck, [3, 4, 6, 3], n, m, is_gray)


# High level feature extractor network (Adopted VGG type structure)
class highLevelNN(nn.Module):
    def __init__(self):
        super(highLevelNN, self).__init__()
        self.CNN = nn.Sequential(
            # first batch (32)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # second batch (64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Third Batch (128)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.CNN(x)
        return out


# Low level feature extraction module
class lowLevelNN(nn.Module):
    def __init__(self):
        super(lowLevelNN, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=256),
            nn.ReLU(),
        )

        # self.conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.fc3 = nn.Linear(in_features=2048, out_features=256)
        # self.fc2 = nn.Linear(in_features=256, out_features=n)
        # self.fc1 = nn.Linear(in_features=n, out_features=m)
        # self.fc0 = nn.Linear(in_features=m, out_features=num_out)

    def forward(self, x):
        x = self.CNN(x)
        # x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=2, padding=1))
        # x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=3, stride=2, padding=1))
        # x = torch.flatten(x, start_dim=1)
        # x = F.relu(self.fc3(x))
        # z_2 = F.relu(self.fc2(x))
        # z_1 = F.relu(self.fc1(z_2))
        # z_0 = self.fc0(z_1)
        return x


class TridentNN(nn.Module):
    def __init__(self, n, m):
        super(TridentNN, self).__init__()
        # Construct the high level neural network
        self.CNN = highLevelNN()
        # Construct the low level neural networks
        self.ageNN = lowLevelNN()  # WARNING: Changed to only age
        # self.genNN = lowLevelNN(num_out=num_gen)
        # self.ethNN = lowLevelNN(num_out=num_eth)
        self.fc2 = nn.Linear(in_features=256, out_features=n)
        self.fc1 = nn.Linear(in_features=n, out_features=m)
        self.fc0 = nn.Linear(in_features=m, out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.CNN(x)
        x = self.ageNN(x)  # z_0 is age
        z_2 = F.relu(self.fc2(x))
        z_1 = F.relu(self.fc1(z_2))
        z_0 = self.fc0(z_1)
        return z_0, z_1, z_2


model_lookup_table = {
    "mlp": MLP,
    "mcmahan_cnn": McMahan_CNN,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "trident": TridentNN,
}

# ############################ Attack ############################ #


def artificial_digits(B, N, nb_digits=0):
    if nb_digits >= 1:
        # adding some fake artificial random digits to B values
        fake_number = [[sum([10**j * randint(0, 9) for j in range(nb_digits)]) for i in range(B.dimensions()[1])]]
        fake_number = matrix(Integers(N), fake_number)
        B_bis = B * 10**nb_digits + fake_number
        return B_bis
    else:
        return B


@hydra.main(config_path="config", config_name="default", version_base="1.1")
def attack(cfg: DictConfig):
    logging.basicConfig(level=cfg.log_level.upper())
    logging.log(INFO, f"Log Level: {cfg.log_level.upper()}")
    setup = {"dtype": torch.float64, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    logging.log(
        INFO,
        f"Running attack {cfg.attack_name=} with batch size={cfg.batch_size} | factor={cfg.factor}"
        + f" | artifcial_digits={cfg.nb_artificial_digits}",
    )
    logging
    logging.log(INFO, f"Dataset: {ID_TO_LABELS[cfg.id]} (id={cfg.id})")
    logging.log(INFO, f"saving path: {os.getcwd()}")
    logging.log(INFO, f"Setup: {setup}")
    logging.log(INFO, f"number of runs: {cfg.nb_runs}")

    for run_id in range(cfg.nb_runs):
        logging.log(
            INFO,
            (
                ("#" * 40)
                + f" {cfg.batch_size=} "
                + f"-- {run_id=}/{cfg.nb_runs} "
                + f"-- dataset={ID_TO_LABELS[cfg.id]}({cfg.id=}) "
                + ("#" * 40)
            ),
        )
        starting_time = time.time()

        all_times = []
        all_current_P = []
        all_results = []
        all_attack_idx = []
        all_e_attacked = []
        all_density_before = []
        all_density_after = []
        all_errors = []
        all_nb_unique_delta = []
        all_B_dimensions = []

        # Dataset
        logging.log(
            INFO,
            f"Loading Dataset: {ID_TO_LABELS[cfg.id]} ({cfg.id=})",
        )
        trainloader, input_shape = get_dataloader(cfg.id, cfg.batch_size, cfg.max_size, cfg.shuffle, setup)

        # Model
        logging.log(INFO, f"Loading Model: {cfg.model_name=}")
        model = get_model(input_shape, cfg.n, cfg.m, cfg.hidden_layer, cfg.model_name).to(**setup)
        logging.log(INFO, f"Model: {model} | {input_shape=} | {cfg.n=} | {cfg.m=}")

        logging.log(INFO, "Computing all vectors")
        all_vectors = get_all_vectors(trainloader, model, cfg.batch_size, testing=cfg.testing, setup=setup)
        all_idx = torch.cat(all_vectors["all_idx"], dim=0).detach().cpu()
        all_delta = torch.cat(all_vectors["all_delta"], dim=0).detach().cpu()
        all_delta_rank = torch.cat([torch.linalg.matrix_rank(d) for d in all_vectors["all_delta"]]).detach().cpu()
        all_e = torch.cat(all_vectors["all_e"], dim=0).cpu()
        all_P = torch.cat(all_vectors["all_P"], dim=0).cpu()
        all_goal = torch.cat(all_vectors["all_goal"], dim=0).cpu()
        all_targets = torch.cat(all_vectors["all_target"], dim=1).transpose(0, 1).cpu()
        all_z_0 = torch.cat(all_vectors["all_z_0"], dim=1).transpose(0, 1).cpu()
        mean_estimated_z_0 = torch.mean(all_z_0.reshape(-1)).detach().cpu()
        std_estimated_z_0 = torch.std(all_z_0.reshape(-1)).detach().cpu()
        logging.log(INFO, f"{mean_estimated_z_0=}, {std_estimated_z_0=}, {all_z_0.shape=}")
        logging.log(INFO, f"{all_P.shape=}, {all_delta.shape=}, {all_e.shape=}, {all_goal.shape=}")
        logging.log(INFO, f"Mean number of 1s in delta: {all_delta.mean().item():.4f}")
        if cfg.testing and cfg.verbose:
            logging.log(INFO, f"{all_delta[0]=}")
            logging.log(INFO, f"{all_P[0]=}")

        max_number = all_P.abs().max().item()

        C = 2**cfg.factor
        logging.log(INFO, f"Computing attack for {len(all_P)} samples")
        for i in range(len(all_P)):
            # if (i + 1) % max((len(all_P) // 100), 1) == 0:  # logging 100 times
            #     logging.log(INFO, f"Progress: {i}/{len(all_P)} " + f"(in {time.time() - starting_time:.2f}s)")
            starting_current_time = time.time()
            attacked_e = np.asarray(all_e[i].sort().values.detach())
            density_before = cfg.batch_size / math.log(max(np.abs(attacked_e)), 2)
            all_density_before.append(density_before)

            N = Integer(hidden_lattice.next_prime(C * 10**cfg.nb_artificial_digits * max_number * 10**6))
            current_P = all_P[i].squeeze()

            # nb_exponent = b2f.get_max_exponent(current_P.reshape(-1))
            # logging.log(INFO, f"{nb_exponent=}")
            nb_exponent = torch.log10(current_P[current_P != 0].abs()).max().item() + 1
            current_P = current_P * 10 ** (-nb_exponent)
            P = C * almost_unique(current_P, treshold=1e-5).detach()
            P = P[P != 0]
            B = matrix(Integers(N), np.int64(np.asarray(P).reshape(1, len(P))))
            B = artificial_digits(B, N, nb_digits=cfg.nb_artificial_digits)
            A = matrix(Integers(N), np.int64(np.asarray(all_delta[i])))

            density_after = cfg.batch_size / math.log(max(P.abs()), 2)
            all_density_after.append(density_after)

            size_L = hidden_lattice.size_basis(matrix(ZZ, np.int64(np.asarray(A))))
            if cfg.verbose:
                logging.log(INFO, f"{attacked_e=}")
                logging.log(INFO, f"{size_L=}")
                logging.log(INFO, f"{current_P=}")
                logging.log(INFO, f"{P=}")
            nb_unique_delta = all_delta[i].unique(dim=0).shape[0]
            all_nb_unique_delta.append(nb_unique_delta)
            all_B_dimensions.append(B.dimensions())
            logging.log(INFO, f"{nb_unique_delta=} | {B.dimensions()=} | {attacked_e.max()=:.4f}")
            try:
                if cfg.attack_name == "ns_original":
                    _, rec_lat, rec, _, rec_noisy = main.recover_hidden_noisy_lattice_algo_II(
                        B, Integer(cfg.batch_size), Integer(len(P)), N, verbose=cfg.verbose
                    )
                    ra, _ = main.ns_original(
                        B, rec, rec_noisy, Integer(cfg.batch_size), Integer(len(P)), N, verbose=cfg.verbose
                    )
                elif cfg.attack_name == "multivariate":
                    _, rec_lat, rec, _, rec_noisy = main.recover_hidden_noisy_lattice_algo_II(
                        B, Integer(cfg.batch_size), Integer(len(P)), N, verbose=cfg.verbose
                    )
                    ra, _ = main.eigen(
                        B, rec_lat, rec_noisy, Integer(cfg.batch_size), Integer(len(P)), N, verbose=cfg.verbose
                    )
                elif cfg.attack_name == "statistical":
                    _, rec_lat, rec, _, rec_noisy = main.recover_hidden_noisy_lattice_algo_II(
                        B, Integer(cfg.batch_size), Integer(len(P)), N, verbose=cfg.verbose
                    )
                    ra, _ = main.statistical(
                        B, rec, rec_noisy, Integer(cfg.batch_size), Integer(len(P)), N, verbose=cfg.verbose
                    )
                if ra[0] == 0:
                    raise ValueError("Error: ra[0] == 0")
                output = (ra[2 : cfg.batch_size + 2] / (-ra[0])).astype(np.int64)
                correct_output = sorted(output / C / 10**cfg.nb_artificial_digits / 10 ** (-nb_exponent))

                if cfg.verbose:
                    logging.log(INFO, f"{correct_output=}")
                    logging.log(INFO, f"{attacked_e=}")
                    ratio = correct_output / attacked_e
                    logging.log(INFO, f"{ratio=}")
                # append results
                all_errors.append([None, None, density_before, density_after])
                all_results.append(correct_output)
                all_attack_idx.append(i)
                all_e_attacked.append(attacked_e)
                all_times.append(time.time() - starting_current_time)
                all_current_P.append(P.detach())
                mae = np.sum(np.abs(correct_output - attacked_e))
                if mae > np.max(np.abs(attacked_e)):
                    logging.log(
                        INFO,
                        f"Error for i={i}(/{len(all_P)}) (SUMMED MAE TOO HIGH: {mae:.4f}"
                        + f"(SUMMED MAE) > {np.max(np.abs(attacked_e))} (MAX))",
                    )
                    # print(f"correct_output: {correct_output}")
                    # print(f"attacked_e: {attacked_e}")
                    if cfg.verbose:
                        logging.log(INFO, f"correct_output: {correct_output}")
                        logging.log(INFO, f"attacked_e: {attacked_e}")
                else:
                    best_mae = torch.mean(
                        torch.abs((torch.tensor(correct_output) + mean_estimated_z_0) - all_targets[i].sort().values)
                    )
                    correct_mae = torch.mean(
                        torch.abs((mean_estimated_z_0 - torch.tensor(correct_output)) - all_targets[i].sort().values)
                    )
                    approx_mae = torch.mean(torch.abs(torch.tensor(correct_output) - all_targets[i].sort().values))
                    logging.log(
                        INFO,
                        f"##### SUCESS(batch_size={cfg.batch_size}), "
                        + f"SUMMED ERROR={mae:.4f} | {best_mae=:.4f} | {approx_mae=:.4F} | {correct_mae=:.4f} "
                        + f"for i={i} (out of {len(all_P)}) "
                        + f"(in {time.time() - starting_current_time:.2f}s) #####",
                    )
            except Exception as e:
                logging.log(INFO, e)
                logging.log(INFO, f"Exception: Error for i={i}")
                all_errors.append([i, e, density_before, density_after])
                all_results.append(None)
                all_attack_idx.append(None)
                all_e_attacked.append(attacked_e)  # not in 08/01/2025 exps
                all_times.append(time.time() - starting_current_time)
                all_current_P.append(P.detach())

            if cfg.testing:
                break

        logging.log(INFO, f"Time for batch size={cfg.batch_size}: {time.time() - starting_time}")
        logging.log(INFO, f"Mean Density Before: {np.mean(all_density_before):.4f}")
        logging.log(
            INFO, f"Mean Density Before over 1/(Batch Size): {np.mean(all_density_before) / cfg.batch_size:.4f}"
        )
        logging.log(INFO, f"Mean Density After: {np.mean(all_density_after):.4f}")
        logging.log(INFO, f"Mean Density After over 1/(Batch Size): {np.mean(all_density_after) / cfg.batch_size:.4f}")
        logging.log(INFO, f"Mean Time: {np.mean(all_times):.4f}")
        logging.log(INFO, f"% Error: {(len(all_errors) / (i+1))*100:.2f}%")
        logging.log(INFO, "\n\n\n")
        dict_to_save = {
            "id": cfg.id,
            "label": ID_TO_LABELS[cfg.id],
            "all_errors": all_errors,
            "all_idx": all_idx,
            "all_K": cfg.batch_size,
            "all_results": all_results,
            "all_P": all_P,
            "all_current_P": all_current_P,
            "all_e_attacked": all_e_attacked,
            "all_delta": all_delta,
            "all_delta_rank": all_delta_rank,
            "all_nb_unique_delta": all_nb_unique_delta,
            "all_B_dimensions": all_B_dimensions,
            "all_attack_idx": all_attack_idx,
            "all_times": all_times,
            "all_goal": all_goal,
            "all_density_before": all_density_before,  # not in 08/01/2025 exps
            "all_density_after": all_density_after,  # not in 08/01/2025 exps
            "all_targets": all_targets,
            "all_z_0": all_z_0,
            "mean_estimated_z_0": mean_estimated_z_0,
            "std_estimated_z_0": std_estimated_z_0,
        }

        np.save(
            f"{os.getcwd()}/attack_results_{run_id}_id={cfg.id}_{cfg.attack_name}_"
            + f"K={cfg.batch_size}_factor={cfg.factor}",
            dict_to_save,
        )
        torch.save(
            model.state_dict(), f"{os.getcwd()}/model_{run_id}_id={cfg.id}_{cfg.attack_name}_K={cfg.batch_size}.pt"
        )

        if cfg.testing:
            break


if __name__ == "__main__":
    OmegaConf.register_new_resolver("my_subdir_suffix", my_subdir_suffix_impl)
    attack()
