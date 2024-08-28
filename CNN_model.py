import torch
import torch.nn as nn
import torch.optim as optim


class CNNModel(nn.Module):

    #dimension: Size of input Vector - Embedding dimensions
    #numFilters: # of patterns for CNN to learn (Can be adjusted)
    #filterSize: Sizes of FilterPattern (Can be adjusted)
    def __init__(self, dimension, numFilters, filterSize):
        super(CNNModel, self).__init__()

        #Puts everything in convultional layer
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=dimension, out_channels=numFilters, kernel_size=f, padding = f//2)
            for f in filterSize
        ])

        self.fc = nn.Linear(numFilters * len(filterSize), numFilters)

    #Passing the TF-IDF embedding Looks for the pattern in input data from the Convolutional Layers
    def forward (self,x):
        x = x.unsqueeze(1).transpose(1,2)
        
        #Apply ReLu activation to each layer
        conv_results = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(conv_results, dim=1)
        x = self.fc(x)
        return x        
    

def train_cnn(description_embeddings):
    dimension = description_embeddings.shape[1]
    numFilters = 50
    filterSize = [3,4,5]

    model = CNNModel(dimension, numFilters, filterSize)

    #Optimizer and Loss Function (Mean Square Error - Used to predict continuous values from the embeddings)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    loss = nn.MSELoss()

    #Embeddings to Tensor through the model
    tensor = description_embeddings.clone().detach().float()
    description_features = model(tensor)

    return model, description_features
