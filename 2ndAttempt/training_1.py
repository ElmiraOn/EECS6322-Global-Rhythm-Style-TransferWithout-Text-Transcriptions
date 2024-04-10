from data_loader import get_loader
from hparams import hparams, hparams_debug_string
import torch
import torch.optim as optim
import torch.nn as nn
from Model1 import training_1 as Model
from torch.utils.data import DataLoader
from utils import sequence_mask
import warnings
warnings.filterwarnings("ignore")


# Define your training loop
def train_model(model1, train_loader, criterion, optimizer, device, epochs):
        model1.train()  # Set the model to training mode
        print("model in training mode 1")
        print("start training...")
        for epoch in range(epochs):
            running_loss = 0.0
            train_iter = iter(train_loader)

            #for data in train_loader:
            for i in range(100):
                try:
                    sp_real, cep_real, cd_real, _, num_rep_sync, len_real, _, len_short_sync, spk_emb = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    sp_real, cep_real, cd_real, _, num_rep_sync, len_real, _, len_short_sync, spk_emb = next(train_iter)

                mask_sp_real = ~sequence_mask(len_real, sp_real.size(1))
                mask_long = (~mask_sp_real).float()
                # for element in data:
                #     print("Element type:", type(element))
                #     print("Element shape:", element.shape if hasattr(element, 'shape') else 'NULL')

                #print(input)
                #return
                codes_mask = sequence_mask(len_short_sync, num_rep_sync.size(1)).float()
                sp_real_sft = torch.zeros_like(sp_real)
                sp_real_sft[:, 1:, :] = sp_real[:, :-1, :]
                len_real_mask = torch.min(len_real + 10, 
                                        torch.full_like(len_real, sp_real.size(1)))
                loss_tx2sp_mask = sequence_mask(len_real_mask, sp_real.size(1)).float().unsqueeze(-1)

                optimizer.zero_grad()
                #outputs = model(cep_real, mask_long)
                spect_pred, stop_pred = model1(cep_real.transpose(2,1),
                                                mask_long,
                                                codes_mask,
                                                num_rep_sync,
                                                len_short_sync+1,
                                                sp_real_sft.transpose(1,0), 
                                                len_real+1,
                                                spk_emb)
                #loss = criterion(stop_pred.squeeze(-1).t(), mask_sp_real.float())
                loss = (criterion(spect_pred.permute(1,0,2), sp_real)
                            * loss_tx2sp_mask).sum() / loss_tx2sp_mask.sum()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                # Save model checkpoints.
            if (i+1) % 10 == 0:
                    torch.save({'model': model1.state_dict(),
                                'optimizer': optimizer.state_dict()}, f'./Model/{i+1}-A.ckpt')
                    print('Saved model1 into Model ...')  

            print(str(epoch) + "--- Done")
        print("Finished Training...")
        print("Start training stage 2")
        




if __name__ == '__main__':
    data_loader = get_loader(hparams)
    print("Loaded data")

    #model = Model(hparams)
    model2 = Model(hparams)
    print("loaded model")
    
    # Define your loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model2.parameters(), lr=0.001)
    # Set the device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model2.to(device)

    # Set number of epochs
    num_epochs = 10

    # train_loader = DataLoader(data_loader, batch_size=32, shuffle=True)
    # Call the training loop
    train_model(model2, data_loader, criterion, optimizer, device, num_epochs)