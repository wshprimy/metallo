import torch
from torch import nn
import torchvision
from torchvision.transforms import ToTensor, Resize, CenterCrop, Normalize, RandomRotation
from torch.utils.data import DataLoader
from torch.utils import data
from tensorboardX import SummaryWriter
from PIL import Image
import os
from torchvision.models import resnet50, resnet18, densenet121, resnet101
import copy
from preprocess import preprocess


def test(test_loader, loss_fn, mynet, device):
    mynet.eval()
    loss = 0
    i = 0
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device).float()
            _, pred = mynet(image)
            pred = torch.flatten(pred)
            loss += loss_fn(pred, label)

            i += 1
        print("loss test is {}".format(loss.item() / i))
    return loss.item() / i


class my_dataset(data.Dataset):
    def __init__(self, image_path_prefix, num, preprocess=False, is_train=True, is_val=False, train_pro=0.8,
                 val_pro=0.1):
        self.process = preprocess
        self.prefix = image_path_prefix
        self.image_path_list = os.listdir(self.prefix)
        self.label_list = [float(x) for x in self.image_path_list]

        self.is_train = is_train
        self.is_val = is_val
        self.train_pro = train_pro
        self.val_pro = val_pro
        self.image_path_list = [os.path.join(self.prefix, x) for x in self.image_path_list]
        print(self.label_list)

        self.num = num
        if self.is_train:
            self.use_num = int(self.num * self.train_pro)
        elif self.is_val:
            self.use_num = int(self.num * self.val_pro)
        else:
            self.use_num = int(self.num * (1 - self.train_pro - self.val_pro))

        self.image_transform = torchvision.transforms.Compose([

            ToTensor(),
            Resize(336),
            CenterCrop(300)
        ])

    def __len__(self):
        return len(self.image_path_list) * self.use_num

    def __getitem__(self, index):
        i = index // self.use_num

        if self.is_train:
            count = index % self.use_num
        elif self.is_val:
            count = index % self.use_num + int(self.num * self.train_pro)
        else:
            count = index % self.use_num + int(self.num * (self.train_pro + self.val_pro))
        image_path = os.path.join(self.image_path_list[i], os.listdir(self.image_path_list[i])[count])
        if self.process:
            image = preprocess(image_path, fill_mode="adaptive")
        else:
            image = Image.open(image_path).convert("RGB")

        label = self.label_list[i]
        out_image = self.image_transform(image)
        if self.is_train or self.is_val:
            return out_image, label
        else:
            return out_image, label, image_path


class my_dataset_save(data.Dataset):
    def __init__(self, image_path_prefix, num, preprocess=False, is_train=True, is_val=False, train_pro=0.8,
                 test_index=[]):
        self.process = preprocess  # 预处理，即去除大噪点
        self.prefix = image_path_prefix
        self.image_path_list = os.listdir(self.prefix)
        self.label_list = [float(x) for x in self.image_path_list]  # 读取文件夹名称，获取DOS
        self.label_list.sort()

        self.test_index = test_index
        self.test_label_list = [self.label_list[x] for x in self.test_index if x < len(self.label_list)]
        self.test_image_path_list = [os.path.join(self.prefix, str(x)) for x in self.test_label_list]

        self.is_train = is_train
        self.is_val = is_val
        self.train_pro = train_pro  # 训练样本所占比例

        self.label_list = [x for x in self.label_list if x not in self.test_label_list]
        self.image_path_list = [os.path.join(self.prefix, str(x)) for x in self.label_list]

        print(self.label_list)
        print(self.image_path_list)
        print(self.test_label_list)
        print(self.test_image_path_list)
        '''for i,label in enumerate(self.label_list):
            image_path=[os.path.join(self.image_path_list[i],x) for x in os.listdir(self.image_path_list[i])]
            random.shuffle(image_path)
            for j,path in enumerate(image_path):
                if j<threshold:
                    train_path.append(path)
                    train_label.append(label)
                else :
                    test_path.append(path)
                    test_label.append(label)'''

        self.num = num  # 计算得出的每组数据数量
        if self.is_train:
            self.use_num = int(self.num * self.train_pro)  # use_num表示当前正在使用的样本数量，按照训练，验证和测试进行了划分。
        elif self.is_val:
            self.use_num = int(self.num * (1 - self.train_pro))
        else:
            self.use_num = self.num

        self.image_transform = torchvision.transforms.Compose([

            ToTensor(),
            # Resize(960),
            # CenterCrop(900),

        ])

    def __len__(self):
        if self.is_train or self.is_val:
            return len(self.image_path_list) * self.use_num  # 这个数值用来限定下面index的最大取值。
        else:
            return len(self.test_label_list) * self.use_num

    def __getitem__(self, index):
        i = index // self.use_num  # i表示当前取的是第几个文件夹

        if self.is_train:
            count = index % self.use_num  # count表示当前取的图片的索引，按照不同情况有所区别
        elif self.is_val:
            count = index % self.use_num + int(self.num * self.train_pro)
        else:
            count = index % self.use_num

        if self.is_train or self.is_val:
            label = self.label_list[i]
            if self.process:
                image = preprocess(os.path.join(self.image_path_list[i], os.listdir(self.image_path_list[i])[count]))
            else:
                image = Image.open(
                    os.path.join(self.image_path_list[i], os.listdir(self.image_path_list[i])[count])).convert("RGB")
        else:
            label = self.test_label_list[i]
            image_path = os.path.join(self.test_image_path_list[i], os.listdir(self.test_image_path_list[i])[count])
            if self.process:
                image = preprocess(image_path)
            else:
                image = Image.open(image_path).convert("RGB")

        out_image = self.image_transform(image)
        if self.is_train or self.is_val:
            return out_image, label
        else:
            return out_image, label, image_path


class CNN(nn.Module):
    def __init__(self, base_net):
        super(CNN, self).__init__()
        self.base = base_net(pretrained=True)

        self.fc = nn.Sequential(
            nn.Linear(1000, 10),
            nn.ReLU(),
            nn.Linear(10, 1)

        )  # (7*7*32,10)

    def forward(self, x):
        x = self.base(x)
        out = nn.Flatten()(x)
        out = self.fc(out)
        return x, out


def train(image_path_prefix, weights_name, data_num, batch_size=4, preprocess=False, base_net=resnet50, device='cpu',
          learn_rate=0.001, weight_dacay=0.00001, epoch=10, save=False, test_part=[]):
    if save:
        train_data = my_dataset_save(image_path_prefix, data_num, preprocess, is_train=True, test_index=test_part)
        val_data = my_dataset_save(image_path_prefix, data_num, preprocess, is_train=False, is_val=True,
                                   test_index=test_part)

        train_loader = DataLoader(train_data, batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size, shuffle=True)

    else:
        train_data = my_dataset(image_path_prefix, data_num, preprocess, is_train=True)
        val_data = my_dataset(image_path_prefix, data_num, preprocess, is_train=False, is_val=True)

        train_loader = DataLoader(train_data, batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size, shuffle=True)

    sum_writer = SummaryWriter()
    '''sum_writer=SummaryWriter(log_dir="C:\\Users\\HP\\Desktop\\test_image1")
    if i%100==0:
        sum_writer.add_scalar("train loss",train_loss.item()/i,global_step=i)

    sum_writer.add_scalar("test accuracy",acc.item(),i)
    sum_writer.add_image("train image",image,i)
    for name,param in mynet.named_parameters():
        sum_writer.add_histogram(name,param.data.numpy(),i)'''

    '''sum_writer=SummaryWriter(log_dir="C:\\Users\\HP\\Desktop\\test_image2")
    history1=hl.History()
    canvas1=hl.Canvas()
    vis=Visdom()'''
    '''train_data=datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )'''
    '''test_data=datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )'''

    '''train_loader=DataLoader(train_data,batch_size=10)'''
    '''test_loader=DataLoader(test_data,batch_size=10)'''

    '''for i,(image,label) in enumerate(train_loader):
        if i==2:
            print(image.shape)
            print(label.shape)
            break

    images=image.squeeze().numpy()
    labels=label.numpy()
    plt.figure(figsize=(8,10))
    for i in range(len(labels)):
        plt.subplot(4,5,i+1)
        plt.imshow(images[i].reshape(28,28),cmap="Blues")
        plt.title(labels[i],size=10)
        plt.axis("off")
        plt.subplots_adjust(wspace=0.05)

    plt.show()'''

    mynet = CNN(base_net).to(device)
    # mynet = DataParallel(mynet, device_ids=[0, 1])
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(mynet.parameters(), lr=learn_rate, weight_decay=weight_dacay)

    '''test(test_loader,loss_fn,mynet,20)'''
    best_loss = 10000
    epoch_num = epoch
    best_model = copy.deepcopy(mynet.state_dict())
    for epoch in range(epoch_num):

        length = len(train_loader)
        loss = 0
        mynet.train()
        for i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device).float()
            _, pred = mynet(image)
            pred = torch.flatten(pred)
            loss = loss_fn(pred, label)
            sum_writer.add_scalar("train loss", loss.item(), global_step=i + epoch * length)
            '''if label.cpu().numpy().mean()<40:
                loss*=1.1'''

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 2 == 0:
                print("training... epoch{}/{}  step{}/{}  loss:{}".format(epoch + 1, epoch_num, i + 1, length,
                                                                          loss.item()))

        if (epoch + 1) % 5 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.96  # 注意这里

        val_loss = test(val_loader, loss_fn, mynet, device)
        sum_writer.add_scalar("val loss", val_loss, global_step=epoch)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(mynet.state_dict())

    mynet.load_state_dict(best_model)
    mynet.eval()
    torch.save(mynet.state_dict(),
               "E:/刘天驰/腐蚀项目/程序/训练程序/pythonProject9_reg_oldserve/pythonProject9_reg_oldserve/weights/" + weights_name + ".pt")

    '''for i,(image,label,image_path) in enumerate(test_loader):
        image=image.to(device)
        label=label.to(device).float()
        feature,pred=mynet(image)
        pred=torch.flatten(pred)
        print(pred-label)
        break

    image_transform=torchvision.transforms.Compose([
            
                Resize((224,224)),
                ToTensor()
            ])
    image_test=Image.open("/home/ubuntu1804/new_jinxiangtu/30.05/image0097.tif")
        
    image_test=image_transform(image_test)
        
    image_test=image_test.unsqueeze(0)
        
    _,pred=mynet(image_test.to(device))
        
    plt.subplot(1,2,1)
    plt.imshow(image_test.squeeze(0).cpu().numpy().transpose(1,2,0))
    plt.title("label:{}".format(30.05))
    plt.subplot(1,2,2)
    plt.imshow(image_test.squeeze(0).cpu().numpy().transpose(1,2,0))
    plt.title("pred:{}".format(pred.detach().cpu().numpy()[0]))
    plt.show()'''


if __name__ == "__main__":
    image_path_prefix = "E:/刘天驰/腐蚀项目/图片集/金相图/9月26(J5083)/金相-5083-20230926"
    train(image_path_prefix, weights_name="1010jinxiang_with_pre_remove(10)", data_num=30, device="cuda",
          base_net=resnet18, epoch=100, batch_size=4, preprocess=True, save=True, test_part=[9])
