from datetime import datetime
import torch
import torch.nn as nn
from torch import autograd
from einops import rearrange
from sklearn.cluster import KMeans
from torchvision import datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import math
import ShapesDataset
from data_utils import NpzDataset
from eval_utils import calc_ari_score

"""
Datasets available:
"2shapes"
"3shapes"
"4Shapes"
"MNIST"
"MNIST_shapes"
"""

N_EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3
SCHEDULER_STEP_SIZE = 10
DATASET = "2shapes"
THRESHOLD = 0.0001
RESUME = False
RESUME_WITH_CUSTOM_LR = False
INFER = False
MODEL_FILE_NAME = "2shapes_take2"
MODEL_PATH = "./models/MNIST_shapes_20240403_213331_12.pt"
PLOT = False


def get_init_bound(fan_in: int):
    return 1 / math.sqrt(fan_in)


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, h_in, dilation=1):
        super().__init__()
        self.layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.normalization = nn.BatchNorm2d(num_features=out_channels)
        
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        
        self.magnitude_bias = nn.Parameter(torch.empty((1, out_channels, 1, 1)))
        nn.init.uniform_(self.magnitude_bias, -get_init_bound(fan_in), get_init_bound(fan_in))

        self.phase_bias = nn.Parameter(torch.empty((1, out_channels, 1, 1)))
        nn.init.constant_(self.phase_bias, val=0)

    def forward(self, x: torch.Tensor):
        return apply_layer(z=x, module=self, normalization=self.normalization)


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, h_in, dilation=1):
        super().__init__()
        self.layer = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            bias=False
        )
        self.normalization = nn.BatchNorm2d(num_features=out_channels)
        
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        
        self.magnitude_bias = nn.Parameter(torch.empty((1, out_channels, 1, 1)))
        nn.init.uniform_(self.magnitude_bias, -get_init_bound(fan_in), get_init_bound(fan_in))

        self.phase_bias = nn.Parameter(torch.empty((1, out_channels, 1, 1)))
        nn.init.constant_(self.phase_bias, val=0)

    def forward(self, x: torch.Tensor):
        return apply_layer(z=x, module=self, normalization=self.normalization)
    
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.layer = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.normalization = nn.BatchNorm1d(num_features=out_features)
        
        fan_in = in_features
        
        self.magnitude_bias = nn.Parameter(torch.empty((1, out_features)))
        nn.init.uniform_(self.magnitude_bias, -get_init_bound(fan_in), get_init_bound(fan_in))

        self.phase_bias = nn.Parameter(torch.empty((1, out_features)))
        nn.init.constant_(self.phase_bias, val=0)

    def forward(self, x: torch.Tensor):
        return apply_layer(z=x, module=self, normalization=self.normalization)


class OutputLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, z: torch.Tensor):
        return nn.functional.sigmoid(self.layer(z.abs()))


def stable_angle(x: torch.tensor, eps=1e-8):
    imag = x.imag
    y = x.clone()
    y.imag[(imag < eps) & (imag > -1.0 * eps)] = eps
    return torch.angle(y)


def apply_layer(z: torch.Tensor, module: nn.Module, normalization):
    psi = torch.complex(module.layer(z.real), module.layer(z.imag))
    synchrony_term = psi.abs() + module.magnitude_bias
    output_phase = stable_angle(psi) + module.phase_bias
    classic_term = module.layer(z.abs()) + module.magnitude_bias
    intermediate_magnitude = 0.5*synchrony_term + 0.5*classic_term
    output_magnitude = nn.functional.relu(normalization(intermediate_magnitude))
    output = torch.complex(output_magnitude*torch.cos(output_phase), output_magnitude*torch.sin(output_phase))
    return output


def preprocess(x_real: torch.Tensor):
    x_imaginary = torch.zeros_like(x_real)
    return x_real * torch.exp(x_imaginary * 1j)


class CAE(nn.Module):
    def __init__(self, number_of_objects: int) -> None:
        super().__init__()
        self.number_of_objects = number_of_objects
        # 32x32 -> 16x16
        self.conv_1 = ComplexConv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1, h_in=32)
        # 16x16 -> 16x16
        self.conv_2 = ComplexConv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, h_in=16)
        # 16x16 -> 8x8
        self.conv_3 = ComplexConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, h_in=16)
        # 8x8 -> 8x8
        self.conv_4 = ComplexConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, h_in=8)
        # 8x8 -> 4x4
        self.conv_5 = ComplexConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, h_in=8)
        self.encoder = nn.Sequential(
            self.conv_1,
            self.conv_2,
            self.conv_3,
            self.conv_4,
            self.conv_5
        )
        self.enc_linear = ComplexLinear(in_features=64*4*4, out_features=512)
        self.dec_linear = ComplexLinear(in_features=512, out_features=64*4*4)
        # 4x4 -> 8x8
        self.conv_t_1 = ComplexConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, h_in=4)
        # 8x8 -> 8x8
        self.conv_d_1 = ComplexConv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, h_in=8)
        # 8x8 -> 16x16
        self.conv_t_2 = ComplexConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1, h_in=8)
        # 16x16 -> 16x16
        self.conv_d_2 = ComplexConv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, h_in=16)
        # 16x16 -> 32x32
        self.conv_t_3 = ComplexConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1, h_in=16)
        self.decoder = nn.Sequential(
            self.conv_t_1,
            self.conv_d_1,
            self.conv_t_2,
            self.conv_d_2,
            self.conv_t_3
        )
        self.output_layer = OutputLayer()
        nn.init.constant_(self.output_layer.layer.weight, 1)
        nn.init.constant_(self.output_layer.layer.bias, 0)

    def evaluate(self, z: torch.Tensor):
        prediction_labels = np.zeros(
            (BATCH_SIZE, 32, 32)
        )
        output = self.output_layer(z).cpu()
        z = z.cpu()
        for image_idx in range(BATCH_SIZE):
            output_magnitude = rearrange(output[image_idx], "c h w -> c (h w)")[0]
            output_magnitude_min = torch.min(output_magnitude, dim=0).values
            output_magnitude_max = torch.max(output_magnitude, dim=0).values
            # Normalization -> [0, 1]
            output_magnitude = torch.div(torch.sub(output_magnitude, output_magnitude_min), torch.sub(output_magnitude_max, output_magnitude_min))
            output_magnitude = torch.unsqueeze(output_magnitude, dim=1)
            background_masking = torch.where(output_magnitude > THRESHOLD, 1.0, 0.0)
            img_phase = z[image_idx].angle()
            img_phase = rearrange(img_phase, "c h w -> (h w) c")
            # Hadamard product
            img_phase = torch.mul(img_phase, background_masking)
            k_means = KMeans(n_clusters=(self.number_of_objects + 1), random_state=0, n_init=10).fit(img_phase)
            cluster_img = rearrange(k_means.labels_, "(h w) -> h w", h=32, w=32)
            prediction_labels[image_idx] = cluster_img

        return prediction_labels

    def forward(self, x: torch.Tensor):
        z = preprocess(x)
        latent = self.encoder(z)
        latent = torch.reshape(latent, (BATCH_SIZE, 64*4*4))
        latent = self.enc_linear(latent)
        latent = self.dec_linear(latent)
        latent = torch.reshape(latent, (BATCH_SIZE, 64, 4, 4))
        output = self.decoder(latent)
        if self.training:
            return self.output_layer(output)
        else:
            return self.output_layer(output), output


def train(train_data_loader, test_data_loader, model, criterion, optimizer, start_epoch, num_epochs, is_MNIST):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outputs = []
    best_loss = 1_000_000.0

    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=SCHEDULER_STEP_SIZE,
            gamma=0.1
        )

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print("================================================")
        print(f'EPOCH {epoch+1}\nLearning rate: {optimizer.param_groups[0]["lr"]}')
        print("------------------------------------------------")
        model.train()
        n = 0
        loss_cumulative_avg = 0
        for (img, _) in train_data_loader:
            # with autograd.detect_anomaly():
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
        if torch.isnan(loss).item():
            raise RuntimeError("Loss returned NaN")
        scheduler.step()
        print(f'Loss: {loss_cumulative_avg:.4f}')
        if loss_cumulative_avg < best_loss:
            best_loss = loss_cumulative_avg
            model_path = './models/{}_{}_{}.pt'.format(MODEL_FILE_NAME, timestamp, epoch+1)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': torch.optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'], weight_decay=1e-5).state_dict(),
                'loss': nn.MSELoss()
                }, model_path)
        outputs.append((epoch, img, recon))

        test(test_data_loader, model, criterion)

        eval(test_data_loader, model, is_MNIST)

    # plot(outputs, num_epochs)


def test(data_loader, model, criterion):
    outputs = []
    model.eval()
    with torch.no_grad():
        n = 0
        loss_cumulative_avg = 0
        for (img, ground_labels) in data_loader:
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

    if PLOT: plot(outputs, 1)


def eval(data_loader, model, is_MNIST, plot:bool = False):
    outputs = []
    model.eval()
    with torch.no_grad():
        for (img, ground_labels) in data_loader:
            img = img.cuda(non_blocking=True)
            recon, output = model(img)
            labels = model.evaluate(output)
            break
        outputs.append((recon, labels))
        if not is_MNIST:
            ari_score = calc_ari_score(batch_size=BATCH_SIZE, labels_true=ground_labels, labels_pred=labels, with_background=False, is_dataset_4Shapes=(DATASET=="4Shapes"))
            print(f'ARI score: {ari_score:.4f}')

    if PLOT or plot: plot_val(outputs)


def plot(image_pairs, num_epochs):
    plt.figure(figsize=(9,2))
    plt.gray()
    images = image_pairs[num_epochs-1][1].cpu().detach().numpy()
    recon = image_pairs[num_epochs-1][2].cpu().detach().numpy()
    for i, item in enumerate(images):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(item[0])
    plt.show()


def plot_val(image_pairs):
    plt.figure(figsize=(9,2))
    plt.gray()
    imgs = image_pairs[0][0].cpu().detach().numpy()
    recon = image_pairs[0][1]
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >= 9: break

        plt.subplot(2, 9, 9+i+1)
        plt.imshow(item)

    plt.show()


def main():
    is_MNIST = False
    match DATASET:
        case "2shapes":
            dataset_train = NpzDataset(dataset_name=DATASET, partition="train")
            dataset_val = NpzDataset(dataset_name=DATASET, partition="val")
            n_objects = 2
        case "3shapes":
            dataset_train = NpzDataset(dataset_name=DATASET, partition="train")
            dataset_val = NpzDataset(dataset_name=DATASET, partition="val")
            n_objects = 3
        case "4Shapes":
            dataset_train = ShapesDataset.ShapesDataset(partition="train")
            dataset_val = ShapesDataset.ShapesDataset(partition="eval")
            n_objects = 4
        case "MNIST_shapes":
            dataset_train = NpzDataset(dataset_name=DATASET, partition="train")
            dataset_val = NpzDataset(dataset_name=DATASET, partition="val")
            n_objects = 2
        case "MNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32,32)),
            ])
            dataset_train = datasets.MNIST(root="./datasets/MNIST/", train=True, transform=transform, download=True)
            dataset_val = datasets.MNIST(root="./datasets/MNIST/", train=False, transform=transform, download=True)
            n_objects = 1
            is_MNIST = True
        case _:
            raise RuntimeError(f'No valid dataset matching the name "{DATASET}"')

    model = CAE(number_of_objects=n_objects)
    model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    epoch = 0

    if RESUME:
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion = checkpoint['loss']
        epoch = checkpoint['epoch'] + 1
    
    if RESUME_WITH_CUSTOM_LR:
        optimizer.param_groups[0]['lr'] = LR
    
    test_data_loader = torch.utils.data.DataLoader(dataset=dataset_val,batch_size=BATCH_SIZE,shuffle=False,drop_last=True)
    if not INFER:
        train_data_loader = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
        train(
            train_data_loader=train_data_loader, 
            test_data_loader=test_data_loader, 
            model=model, 
            criterion=criterion, 
            optimizer=optimizer, 
            start_epoch=epoch, 
            num_epochs=N_EPOCHS,
            is_MNIST=is_MNIST
        )
    
    eval(data_loader=test_data_loader, model=model, is_MNIST=is_MNIST, plot=True)


if __name__ == "__main__":
    main()
