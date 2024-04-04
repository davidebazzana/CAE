import torch
import torch.nn as nn
from data_utils import NpzDataset
import sys
from getopt import GetoptError, getopt

from CAE import CAE
from train_utils import train, eval

"""
Datasets available:
"2shapes"
"3shapes"
"4Shapes"
"MNIST_shapes"
"""

BATCH_SIZE = 64
LR = 1e-3


def main():
    try:
        opts, args = getopt(sys.argv[1:], "", ["task=", "model=", "dataset=", "epochs="])
    except GetoptError:
        print("Wrong arguments")

    model = CAE()
    model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    epoch = 0

    opts = dict(opts)

    dataset = opts["--dataset"]
    match dataset:
        case "2shapes":
            dataset_train = NpzDataset(dataset_name=dataset, partition="train")
            dataset_val = NpzDataset(dataset_name=dataset, partition="val")
            n_objects = 2
        case "3shapes":
            dataset_train = NpzDataset(dataset_name=dataset, partition="train")
            dataset_val = NpzDataset(dataset_name=dataset, partition="val")
            n_objects = 3
        case "MNIST_shapes":
            dataset_train = NpzDataset(dataset_name=dataset, partition="train")
            dataset_val = NpzDataset(dataset_name=dataset, partition="val")
            n_objects = 2
        case _:
            raise RuntimeError(f'No valid dataset matching the name "{dataset}"')
    test_data_loader = torch.utils.data.DataLoader(dataset=dataset_val,batch_size=BATCH_SIZE,shuffle=False,drop_last=True)

    match opts["--task"]:
        case "train":
            epochs = int(opts["--epochs"])

            train_data_loader = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
            train(
                train_data_loader=train_data_loader, 
                test_data_loader=test_data_loader, 
                model=model, 
                criterion=criterion, 
                optimizer=optimizer, 
                start_epoch=epoch, 
                num_epochs=epochs,
                number_of_objects=n_objects,
                model_file_name=dataset,
            )
        case "evaluate":
            model_path = opts["--model"]

            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            criterion = checkpoint['loss']
            epoch = checkpoint['epoch'] + 1
            
            eval(data_loader=test_data_loader, model=model, number_of_objects=n_objects, plot=True)


if __name__ == "__main__":
    main()
