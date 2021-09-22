import torch
from torch._C import dtype
import torch.nn as nn
import torch.optim as optim

def train_gan(G, D, dataloader, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'torch device : {device}')
    
    G.to(device)
    D.to(device)

    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = optim.Adam(D.parameters(), d_lr, [beta1, beta2])

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    z_dim = 20

    batch_size = dataloader.batch_size

    for epoch in range(num_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for imgs in dataloader:
            if imgs.size()[0] == 1:
                continue

            # Discriminator
            imgs = imgs.to(device)
            mini_batch_size = imgs.size()[0]
            label_real = torch.full((mini_batch_size,), 1, dtype=torch.float32).to(device)
            label_fake = torch.full((mini_batch_size,), 0, dtype=torch.float32).to(device)

            # real image
            d_out_real, _ = D(imgs)

            # fake image
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_imgs = G(input_z)
            d_out_fake, _  = D(fake_imgs)

            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            # Generator
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake, _ = D(fake_images)

            g_loss = criterion(d_out_fake.view(-1), label_real)
            
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            epoch_d_loss += float(d_loss)
            epoch_g_loss += float(g_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}')
            print(f'D_Loss:{epoch_d_loss/batch_size:.4f}, G_Loss:{epoch_g_loss/batch_size:.4f}')
            
    return G, D
