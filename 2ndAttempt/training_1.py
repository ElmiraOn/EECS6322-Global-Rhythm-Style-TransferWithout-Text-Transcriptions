from data_loader import get_loader
from hparams_autopst import hparams, hparams_debug_string
import torch
import torch.optim as optim
import torch.nn as nn
from Model import Model 
from torch.utils.data import DataLoader
from utils import sequence_mask
import warnings
warnings.filterwarnings("ignore")

# Define your training loop
def train_model(model, train_loader, criterion, optimizer, device, epochs):
        model.train()  # Set the model to training mode
        print("model in training mode")
        print("start training...")
        for epoch in range(epochs):
            running_loss = 0.0
            #for data in train_loader:

            train_iter = iter(train_loader)

            try:
                sp_real, cep_real, cd_real, _, num_rep_sync, len_real, _, len_short_sync, spk_emb = next(data_iter)
            except:
                data_iter = iter(data_loader)
                sp_real, cep_real, cd_real, _, num_rep_sync, len_real, _, len_short_sync, spk_emb = next(data_iter)

            mask_sp_real = ~sequence_mask(len_real, sp_real.size(1))
            mask_long = (~mask_sp_real).float()
            # for element in data:
            #     print("Element type:", type(element))
            #     print("Element shape:", element.shape if hasattr(element, 'shape') else 'NULL')

            #print(input)
            #return

            optimizer.zero_grad()
            outputs = model(cep_real, mask_long)
            loss = criterion(outputs.squeeze(-1).t(), mask_sp_real.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(str(epoch) + "--- Done")
        print("Finished Training...")


if __name__ == '__main__':
    data_loader = get_loader(hparams)
    print("Loaded data")

    model = Model(hparams)
    print("loaded model")
    
    # Define your loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Set the device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model.to(device)

    # Set number of epochs
    num_epochs = 10

    # train_loader = DataLoader(data_loader, batch_size=32, shuffle=True)
    # Call the training loop
    train_model(model, data_loader, criterion, optimizer, device, num_epochs)
