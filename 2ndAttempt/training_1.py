from data_loader import get_loader
from hparams_autopst import hparams, hparams_debug_string
import torch
import torch.optim as optim
import torch.nn as nn
from Model import Model 
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

# Define your training loop
def train_model(model, train_loader, criterion, optimizer, device, epochs):
        model.train()  # Set the model to training mode
        print("model in training mode")
        print("start training...")
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, data in enumerate(train_loader):
                print(len(data))
        
            print(str(epoch) + "--- Done")
        print("Finished Training...")


if __name__ == '__main__':
    data_loader = get_loader(hparams)
    print("Loaded data")

    model = Model(hparams)
    print("loaded model")
    
    # # Define your loss function and optimizer
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # # Set the device to GPU if available, else CPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # Move the model to the device
    # model.to(device)


    # # Set number of epochs
    # num_epochs = 10

    # # train_loader = DataLoader(data_loader, batch_size=32, shuffle=True)
    # # Call the training loop
    # train_model(model, data_loader, criterion, optimizer, device, num_epochs)
