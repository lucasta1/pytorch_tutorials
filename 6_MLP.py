import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from torchvision import datasets, transforms

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.linear1 = Dense(in_dim, hid_dim)
        self.linear2 = Dense(hid_dim, out_dim)

    def forward(self, x):
        x = relu(self.linear1(x))
        x = softmax(self.linear2(x))
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_dim = 784
hid_dim = 200
out_dim = 10
lr = 0.001
batch_size = 32
n_epochs = 10


mlp = MLP(in_dim, hid_dim, out_dim).to(device)

optimizer = optim.SGD(mlp.parameters(), lr=lr)

# 前処理を定義
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(in_dim))
])

# torchvisionのdatasetsを使ってMNISTのデータを取得
# ミニバッチ化や前処理などの処理を行ってくれるDataLoaderを定義
dataloader_train = torch.utils.data.DataLoader(
    datasets.MNIST('./data/mnist', train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)

dataloader_valid = torch.utils.data.DataLoader(
    datasets.MNIST('./data/mnist', train=False, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=False
)

for epoch in range(n_epochs):
    losses_train = []
    losses_valid = []
    train_num = 0
    train_true_num = 0
    valid_num = 0
    valid_true_num = 0

    mlp.train()  # 訓練時には勾配を計算するtrainモードにする
    for x, t in dataloader_train:
        true = t.tolist()

        t_hot = torch.eye(10)[t]  # 正解ラベルをone-hot vector化

        # 勾配の初期化
        mlp.zero_grad()

        # テンソルをGPUに移動
        x = x.to(device)
        t_hot = t_hot.to(device)

        # 順伝播
        y = mlp.forward(x)

        # 誤差の計算(クロスエントロピー誤差関数)
        loss = -(t_hot*torch.log(y)).sum(axis=1).mean()

        # 誤差の逆伝播
        optimizer.zero_grad()
        loss.backward()

        # パラメータの更新
        optimizer.step()

        # モデルの出力を予測値のスカラーに変換
        pred = y.argmax(1)

        losses_train.append(loss.tolist())

        acc = torch.where(t - pred.to("cpu") == 0, torch.ones_like(t), torch.zeros_like(t))
        train_num += acc.size()[0]
        train_true_num += acc.sum().item()

    mlp.eval()  # 評価時には勾配を計算しないevalモードにする
    for x, t in dataloader_valid:
        true = t.tolist()

        t_hot = torch.eye(10)[t]  # 正解ラベルをone-hot vector化

        # テンソルをGPUに移動
        x = x.to(device)
        t_hot = t_hot.to(device)

        # 順伝播
        y = mlp.forward(x)

        # 誤差の計算(クロスエントロピー誤差関数)
        loss = -(t_hot*torch.log(y)).sum(axis=1).mean()

        # モデルの出力を予測値のスカラーに変換
        pred = y.argmax(1)

        losses_valid.append(loss.tolist())

        acc = torch.where(t - pred.to("cpu") == 0, torch.ones_like(t), torch.zeros_like(t))
        valid_num += acc.size()[0]
        valid_true_num += acc.sum().item()

    print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
        epoch,
        np.mean(losses_train),
        train_true_num/train_num,
        np.mean(losses_valid),
        valid_true_num/valid_num
    ))