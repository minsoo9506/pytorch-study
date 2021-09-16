import torch

def train_model(net, dataloader_dict, crit, optimizer, num_epoch, mode, device):

    net.to(device)

    for epoch in range(num_epoch):
        print(f'Epoch {epoch + 1}')

        epoch_loss = 0.0
        epoch_correct = 0

        if mode == 'train':
            net.train()
            for inputs, labels in dataloader_dict['train']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                loss = crit(outputs, labels)
                _, preds = torch.max(outputs, 1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                epoch_correct += torch.sum(preds == labels)
        else:
            net.eval()
            with torch.no_grad():
                for inputs, labels in dataloader_dict['val']:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    loss = crit(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_correct += torch.sum(preds == labels)

        epoch_loss /= len(dataloader_dict[mode].dataset)
        epoch_acc = epoch_correct / len(dataloader_dict[mode].dataset)
        print(f'{mode} Loss {epoch_loss}, Acc {epoch_acc}')
        print('------------------------------------------')