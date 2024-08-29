import torch
import torch.nn as nn
import torch.optim as optim


class LSTMModel(nn.Module):

    def __init__(self, inputDim, hiddenDim, outputDim, numLayers):
        """
        Parameters:
        - inputDim: # of input features (size of embedding vector)
        - hiddenDim: # of hidden units in the LSTM
        - outputDim: # of output units (e.g., 1 for regression tasks)
        - numLayers: #  of stacked LSTMs
        """

        super(LSTMModel, self).__init__()

        #Puts everything in a linear layer
        self.lstm = nn.LSTM(inputDim, hiddenDim, numLayers, batch_first= True)

        #Map the hidden layer to the output layer
        self.fc = nn.Linear(hiddenDim, outputDim)
        

    def forward (self,x):
        lstm_out, _ = self.lstm(x)
        
        # Take the final hidden state from the output sequence
        lstm_out = lstm_out[:, -1, :]

        return self.fc (lstm_out)
     
    

def train_LSTM(description_embeddings, num_epochs=10, learning_rate=1e-3):
    
    #Convert 2 dimension to 3 dimesnions
    description_embeddings = description_embeddings.unsqueeze(1)

    #Description Embedding: (batch_size, sequence_length, embedding_dim)
    inputDim  = description_embeddings.shape[2]
    hiddenDim = 50
    outputDim = 50   #For regression task
    
    model = LSTMModel(inputDim, hiddenDim, outputDim, numLayers= 1)

    print(f"LSTM ARCHITECTURE {model}\n\n")

    #Optimizer and Loss Function (Mean Square Error - Used to predict continuous values from the embeddings)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    losses = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(description_embeddings)
        target = torch.zeros_like(outputs)
        loss = losses(outputs, target)
        loss.backward()
        optimizer.step()

        print(f'LSTM TRAINING: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # Extract features using the trained model
    with torch.no_grad():
        description_features = model(description_embeddings)

    return model, description_features
