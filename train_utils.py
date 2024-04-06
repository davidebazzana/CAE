from datetime import datetime
import torch
import torch.nn as nn

from plot_utils import plot_pair
from plot_utils import plot_in_out_labels
from eval_utils import calc_ari_score
from eval_utils import object_discovery

def train(train_data_loader, test_data_loader, model, criterion, optimizer, start_epoch, num_epochs, number_of_objects, model_file_name, with_ari_bg, cpu: bool = False):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outputs = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print("================================================")
        print(f'EPOCH {epoch+1}\nLearning rate: {optimizer.param_groups[0]["lr"]}')
        print("------------------------------------------------")
        model.train()
        n = 0
        loss_cumulative_avg = 0
        for (img, _) in train_data_loader:
            # with autograd.detect_anomaly():
            if not cpu:
                img = img.cuda(non_blocking=True)
            recon = model(img)
            loss = criterion(recon, img)

            if n == 0:
                loss_cumulative_avg = loss.item()
            else:
                loss_cumulative_avg = (loss.item() + n*loss_cumulative_avg)/(n+1)
            n += 1

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
        parameters = []
        for p in model.parameters():
            parameters.append(p.norm().item())
        parameters = torch.Tensor(parameters)
        print(f'Loss: {loss_cumulative_avg:.4f}')
        outputs.append((epoch, img, recon))

        test(test_data_loader, model, criterion, cpu=cpu)

        eval(test_data_loader, model, number_of_objects, with_background=with_ari_bg, cpu=cpu)
        
    model_path = './models/{}_{}_{}.pt'.format(model_file_name, timestamp, epoch+1)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': torch.optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'], weight_decay=1e-5).state_dict(),
        'loss': nn.MSELoss()
    }, model_path)


def test(data_loader, model, criterion, cpu: bool = False):
    outputs = []
    model.eval()
    with torch.no_grad():
        n = 0
        loss_cumulative_avg = 0
        for (img, ground_labels) in data_loader:
            if not cpu:
                img = img.cuda(non_blocking=True)
            recon, output = model(img)
            loss = criterion(recon, img)

            if n == 0:
                loss_cumulative_avg = loss.item()
            else:
                loss_cumulative_avg = (loss.item() + n*loss_cumulative_avg)/(n+1)
            n += 1
        outputs.append((1, img, recon))
        print(f'Validation loss: {loss_cumulative_avg:.4f}')


def eval(data_loader, model, number_of_objects, plot:bool = False, with_background=False, cpu: bool = False):
    outputs = []
    model.eval()
    with torch.no_grad():
        for (img, ground_labels) in data_loader:
            if not cpu:
                img = img.cuda(non_blocking=True)
            recon, output = model(img)
            labels = object_discovery(model, output, number_of_objects)
            break
        outputs.append((img.cpu().detach().numpy(), recon.cpu().detach().numpy(), labels))
        ari_score = calc_ari_score(labels_true=ground_labels, labels_pred=labels, with_background=with_background)
        print(f'ARI score: {ari_score:.4f}')

    if plot: plot_in_out_labels(outputs)
