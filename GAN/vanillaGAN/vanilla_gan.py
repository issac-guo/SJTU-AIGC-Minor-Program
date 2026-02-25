import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time  # 用于统计时间

# 超参数设置
batch_size = 128
latent_dim = 100
epochs = 50
learning_rate = 0.0002

# 设置设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 将像素值归一化到[-1, 1]
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义生成器
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# 初始化生成器和判别器
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# 定义损失函数和优化器
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# 用于保存每个 epoch 的损失值
g_losses = []
d_losses = []

# 训练循环
for epoch in range(epochs):
    start_time = time.time()  # 记录当前 epoch 的开始时间

    epoch_g_loss = 0
    epoch_d_loss = 0

    for i, (imgs, _) in enumerate(train_loader):

        # 将数据移动到设备
        imgs = imgs.to(device)
        real = torch.ones(imgs.size(0), 1).to(device)
        fake = torch.zeros(imgs.size(0), 1).to(device)

        # 训练判别器
        optimizer_D.zero_grad()

        # 真实图像的损失
        real_loss = adversarial_loss(discriminator(imgs), real)
        # 生成假图像
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        gen_imgs = generator(z)
        # 假图像的损失
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        # 总损失
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()

        # 生成假图像并计算损失
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), real)
        g_loss.backward()
        optimizer_G.step()

        # 累加每个 batch 的损失值
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()

        # 打印训练信息
        if i % 100 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_loader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    # 计算当前 epoch 的平均损失值
    epoch_g_loss /= len(train_loader)
    epoch_d_loss /= len(train_loader)
    g_losses.append(epoch_g_loss)
    d_losses.append(epoch_d_loss)

    # 计算当前 epoch 的用时
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")
    print(f"Average G loss: {epoch_g_loss:.4f}, Average D loss: {epoch_d_loss:.4f}")

    # 每 5 个 epoch 保存生成的图像
    if epoch % 5 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim).to(device)
            gen_imgs = generator(z)
            gen_imgs = 0.5 * gen_imgs + 0.5  # 反归一化
            gen_imgs = gen_imgs.cpu().numpy()
            fig, axs = plt.subplots(4, 4, figsize=(4, 4))
            cnt = 0
            for i in range(4):
                for j in range(4):
                    axs[i, j].imshow(gen_imgs[cnt, 0, :, :], cmap='gray')
                    axs[i, j].axis('off')
                    cnt += 1
            plt.suptitle(f"Epoch {epoch}\nD loss: {epoch_d_loss:.4f}, G loss: {epoch_g_loss:.4f}")
            plt.savefig(f"gan_generated_epoch_{epoch}.png")
            plt.close()

# 训练完成后保存模型
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
print("Training complete and models saved.")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label="Generator Loss")
plt.plot(d_losses, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Generator and Discriminator Loss During Training")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()