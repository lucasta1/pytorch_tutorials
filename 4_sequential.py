import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

rng = np.random.RandomState(1234)
random_state = 42


def relu(x):
    x = torch.where(x > 0, x, torch.zeros_like(x))
    return x


def softmax(x):
    x -= torch.cat([x.max(axis=1, keepdim=True).values] * x.size()[1], dim=1)
    x_exp = torch.exp(x)
    return x_exp / torch.cat([x_exp.sum(dim=1, keepdim=True)] * x.size()[1], dim=1)

class Dense(nn.Module):  # nn.Moduleを継承する
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        super().__init__()
        # He Initialization
        # in_dim: 入力の次元数、out_dim: 出力の次元数
        self.W = nn.Parameter(torch.tensor(rng.uniform(
            low=-np.sqrt(6 / in_dim),
            high=np.sqrt(6 / in_dim),
            size=(in_dim, out_dim)
        ).astype('float32')))
        self.b = nn.Parameter(torch.tensor(np.zeros([out_dim]).astype('float32')))
        self.function = function
    
    def forward(self, x):  # forwardをoverride
        return self.function(torch.matmul(x, self.W) + self.b)


class MLP(nn.Module):  # nn.Moduleを継承する
    def __init__(self, in_dim, hid_dim, out_dim):  # __init__をoverride
        super(MLP, self).__init__()
        self.linear1 = Dense(in_dim, hid_dim)
        self.linear2 = Dense(hid_dim, out_dim)
    
    def forward(self, x):  # forwardをoverride
        x = relu(self.linear1(x))
        x = softmax(self.linear2(x))
        return x

mlp = nn.Sequential(
    Dense(2, 3, relu),  # 自分て定義した全結合層を重ねて2層ネットワークを定義する
    Dense(3, 2, softmax)
)

# mlp = MLP(2, 3, 2) でも同様のネットワークを定義できる

print(mlp)
print()

x = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = mlp(x)  # forward(x)が呼ばれる
print("# feedforward：")
print(y)
print()

print("# mlp.parameters()でモデルのパラメータ取得：")
print(mlp.parameters())

# XORをMLPで行う
x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
t = torch.tensor([0, 1, 1, 0], dtype=torch.long)

# モデルの定義
mlp = MLP(2, 3, 2)

# 最適化の定義
optimizer = optim.SGD(mlp.parameters(), lr=0.1)  # Moduleのパラメータは.parameters()で取得できる

# モデルを訓練モードにする（Dropout等に関係）
mlp.train()

for i in range(1000):

    t_hot = torch.eye(2)[t]  # 正解ラベルをone-hot vector化

    # 順伝播
    y_pred = mlp(x)

    # 誤差の計算(クロスエントロピー誤差関数)
    loss = -(t_hot*torch.log(y_pred)).sum(axis=1).mean()

    # 逆伝播
    optimizer.zero_grad()
    loss.backward()

    # パラメータの更新
    optimizer.step()

    if i % 100 == 0:
        print(i, loss.item())
    
print(list(mlp.parameters()))
print()

# state_dictの取得
state_dict = mlp.state_dict()
print(state_dict)

# モデルの保存
torch.save(state_dict, './model.pth')

# モデルの定義
mlp2 = MLP(2, 3, 2)
print(list(mlp2.parameters()))  # ランダムな初期値
print()

# 学習済みパラメータの読み込み
state_dict = torch.load('./model.pth')
mlp2.load_state_dict(state_dict)
print(list(mlp2.parameters()))  # 学習済みパラメータ