import torch
from torch import nn
import torchvision
from torchvision.transforms import ToTensor, Resize, CenterCrop
from torch.utils.data import DataLoader
from torch.utils import data
from tensorboardX import SummaryWriter
import pandas as pd
import os
import copy
import numpy as np


class my_dataset(data.Dataset):
    def __init__(self, image_path_prefix, mode="sum", sample="1.5-0.1", is_train=True, is_val=False, train_pro=0.8,
                 val_pro=0.1):

        self.train_data_list = []
        self.val_data_list = []
        self.test_data_list = []
        self.mode = mode
        self.prefix = image_path_prefix
        self.dos_path_list = os.listdir(self.prefix)

        self.label_list = sorted([float(x.split(".")[0] + "." + x.split(".")[1]) for x in self.dos_path_list])
        print(self.label_list)
        # self.image_path_list=[os.path.join(self.prefix,x,sample) for x in self.dos_path_list]
        self.train_pro = train_pro
        self.val_pro = val_pro
        self.is_train = is_train
        self.is_val = is_val

        for i in range(len(self.label_list)):

            frame = pd.read_csv(os.path.join(self.prefix, self.dos_path_list[i]))
            frame = np.transpose(np.array(frame))
            frame = frame.tolist()

            if i == 0:
                self.num = len(frame)

            temp_train_data_list = frame[0:int(len(frame) * self.train_pro)]
            for sub_list in temp_train_data_list:
                sub_list += [self.label_list[i]]
            self.train_data_list += temp_train_data_list
            sub_list = []

            temp_val_data_list = frame[
                                 int(len(frame) * self.train_pro):int(len(frame) * (self.train_pro + self.val_pro))]
            for sub_list in temp_val_data_list:
                sub_list += [self.label_list[i]]
            self.val_data_list += temp_val_data_list
            sub_list = []

            temp_test_data_list = frame[int(len(frame) * (self.train_pro + self.val_pro)):len(frame)]
            for sub_list in temp_test_data_list:
                sub_list += [self.label_list[i]]
            self.test_data_list += temp_test_data_list

        if self.is_train:
            self.use_num = int(self.num * self.train_pro)
        elif self.is_train:
            self.use_num = int(self.num * self.val_pro)
        else:
            self.use_num = int(self.num * (1 - self.train_pro - self.val_pro))

        self.image_transform = torchvision.transforms.Compose([
            ToTensor()
        ])

    def __len__(self):
        return len(self.dos_path_list) * self.use_num

    def __getitem__(self, index):

        if self.is_train:
            data = self.train_data_list[index]
        elif self.is_val:
            data = self.val_data_list[index]
        else:
            data = self.test_data_list[index]
        new_data = np.array(data)
        new_data / max(new_data[0:len(new_data) - 1])
        return torch.tensor(new_data[0:len(data) - 1]), data[len(data) - 1]


class my_dataset_save(data.Dataset):
    def __init__(self, image_path_prefix, is_train=True, is_val=False, train_pro=0.8, val_pro=0.2, save_num=[]):

        self.train_data_list = []
        self.val_data_list = []
        self.test_data_list = []

        self.prefix = image_path_prefix
        self.dos_path_list = os.listdir(self.prefix)
        self.dos_path_list = sorted([float(x.split(".")[0] + "." + x.split(".")[1]) for x in self.dos_path_list])
        self.train_pro = train_pro
        self.val_pro = val_pro
        self.is_train = is_train
        self.is_val = is_val
        if self.is_train or self.is_val:
            self.label_list = [self.dos_path_list[x] for x in range(len(self.dos_path_list)) if x not in save_num]
            test_dos = [self.dos_path_list[x] for x in range(len(self.dos_path_list)) if x in save_num]
        else:
            self.label_list = [self.dos_path_list[x] for x in range(len(self.dos_path_list)) if x in save_num]

        print(self.label_list)
        if self.is_train:
            print(test_dos)
        # self.image_path_list=[os.path.join(self.prefix,x,sample) for x in self.dos_path_list]

        for i in range(len(self.label_list)):

            frame = pd.read_csv(os.path.join(self.prefix, str(self.label_list[i]) + ".csv"), header=None)
            frame = np.transpose(np.array(frame))
            frame = frame.tolist()

            if i == 0:
                self.num = len(frame)  # 每组数据个数
            if self.is_train:
                temp_train_data_list = frame[0:int(len(frame) * self.train_pro)]
                for sub_list in temp_train_data_list:
                    sub_list += [self.label_list[i]]
                self.train_data_list += temp_train_data_list
                sub_list = []
            elif self.is_val:
                temp_val_data_list = frame[int(len(frame) * self.train_pro):int(len(frame))]
                for sub_list in temp_val_data_list:
                    sub_list += [self.label_list[i]]
                self.val_data_list += temp_val_data_list
                sub_list = []
            else:
                temp_test_data_list = frame[0:len(frame)]
                for sub_list in temp_test_data_list:
                    sub_list += [self.label_list[i]]
                self.test_data_list += temp_test_data_list

        if self.is_train:
            self.use_num = int(self.num * self.train_pro)
        elif self.is_val:
            self.use_num = int(self.num * self.val_pro)
        else:
            self.use_num = int(self.num)

        self.image_transform = torchvision.transforms.Compose([
            ToTensor()
        ])

    def __len__(self):
        return len(self.label_list) * self.use_num

    def __getitem__(self, index):

        if self.is_train:
            data = self.train_data_list[index]
        elif self.is_val:
            data = self.val_data_list[index]
        else:
            data = self.test_data_list[index]
        new_data = np.array(data)
        new_data / max(new_data[0:len(new_data) - 1])
        return torch.tensor(new_data[0:len(data) - 1]), data[len(data) - 1]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.att = nn.Linear(1600, 1600)
        self.fc = nn.Sequential(
            nn.Linear(1600, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU(),
            nn.Linear(2000, 5000),
            nn.BatchNorm1d(5000),
            nn.ReLU(),
            nn.Linear(5000, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU(),
            nn.Linear(2000, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 1),

        )

    def forward(self, x):
        atten = self.att(x)
        temp = x * atten
        out = self.fc(temp)

        return out


def test(test_loader, loss_fn, mynet, device="cuda"):
    mynet.eval()
    loss = 0
    i = 0
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device).float()
            label = label.to(device).float()
            image = image.squeeze(1)
            pred = mynet(image)
            pred = torch.flatten(pred)
            loss += loss_fn(pred, label)
            i += 1
        print("loss test is {}".format(loss.item() / i))
    return loss.item() / i


'''test(test_loader,loss_fn,mynet,20)'''


def train(path_prefix, weight_name, train_p=0.8, val_p=0.2, device="cuda", is_save=False, save=[], epoch=100,
          batch_s=64):
    if is_save:
        train_data = my_dataset_save(path_prefix, train_pro=train_p, save_num=save)
        test_data = my_dataset_save(path_prefix, is_train=False, is_val=True, train_pro=train_p, val_pro=val_p,
                                    save_num=save)
        train_loader = DataLoader(train_data, batch_size=batch_s, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_s, shuffle=True)
    else:
        train_data = my_dataset(path_prefix, train_pro=train_p)
        test_data = my_dataset(path_prefix, is_train=False, is_val=True, train_pro=train_p, val_pro=val_p)
        train_loader = DataLoader(train_data, batch_size=batch_s, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_s, shuffle=True)

    best_loss = 10000
    mynet = CNN().to(device)
    loss_fn = nn.MSELoss(reduction="mean")
    sum_writer = SummaryWriter()

    optimizer = torch.optim.Adam(mynet.parameters(), lr=0.01, weight_decay=0.00001)
    epoch_num = epoch
    best_model = copy.deepcopy(mynet.state_dict())
    for epoch in range(epoch_num):

        length = len(train_loader)
        loss = 0
        mynet.train()
        for i, (image, label) in enumerate(train_loader):
            image = image.to(device).float()
            label = label.to(device).float()
            image = image.squeeze(1)

            pred = mynet(image)
            pred = torch.flatten(pred)
            loss = loss_fn(pred, label)
            sum_writer.add_scalar("train loss", loss.item(), global_step=i + epoch * length)

            symbol = epoch * length + i + 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 5 == 0:
                print("training... epoch{}/{}  step{}/{}  loss:{}".format(epoch + 1, epoch_num, i + 1, length,
                                                                          loss.item()))

        if (epoch + 1) % 10 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.92  # 注意这里

        val_loss = test(test_loader, loss_fn, mynet)
        sum_writer.add_scalar("val loss", val_loss, global_step=epoch)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(mynet.state_dict())

    mynet.load_state_dict(best_model)
    mynet.eval()
    torch.save(mynet.state_dict(), r"D:\HJF\测试demo\LIBS\weights" + "\\" + weight_name)


if __name__ == "__main__":
    train(r"D:\libsdata\20250514-L5083-Spec\FirstMovingAverage",
          "paper20250514-L5083-FirstMovingAverage-2-libsdata-sum_600epoch_save().pt",
          is_save=True, save=[], epoch=600, batch_s=256)
