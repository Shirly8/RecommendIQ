import torch
import torch.nn as nn
import torch.optim as optim
from Recommend import loadData, convertCategories, convertDescriptions
from Cnn_model import CNNModel, train_cnn_model

class NCFModel(nn.Module):

    def __init__(self, numUsers, numItems, dimensions, featuresDimensions):
        super(NCFModel, self).__init__()


        #Create embeddings for users and items
        self.user_embedding = nn.Embedding(numUsers, dimensions)
        self.item_embedding = nn.Embedding(numItems, dimensions)

        #Combine users and items with descriptions CNNmodel into single vector of 64 to predict scores
        self.fc1 = nn.Linear(dimensions*2 + featuresDimensions, 64)
        self.fc2 = nn.Linear(64, 1) #Gives score on how much user likes item


        
    def forward(self, userID, itemID, features):
        userEmbedding = self.user_embedding(userID)
        itemEmbedding = self.user_embedding(itemID)

        #Passing through the layer 
        x = torch.cat([userEmbedding, itemEmbedding, features], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

    def train_ncf(userData, features): 
        numUsers = userData['User_ID'].nunique()
        numItems = userData['Item_ID'].nunique()
        dimensions = 20
        numFilters = 50

        #Optimizer and Loss Function (Mean Square Error - Used to predict continuous values from the embeddings)
        model = NCFModel(numUsers, numItems, dimensions, numFilters)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        losses = nn.MSELoss()

        #Embeddings to Tensor through the model
        userID = torch.tensor(userData['User_ID'].values, dtype=torch.long)
        itemID = torch.tensor(userData['Item_ID'].values, dtype=torch.long)
        ratings = torch.tensor(userData['Rating'].values, dtype=torch.float32)

        NCFModel.train()

        for epoch in range(10):
            optimizer.zero_grad()  #Clears previous gradients to prevent accumulation
            prediction = NCFModel(userID, itemID, features)
            loss = losses(prediction.squeeze(), ratings)
            loss.backward() #Compute gradients
            optimizer.step()

            print(f'Epoch {epoch+1}/{10}, Loss: {loss.item()}')

            torch.save(NCFModel.state_dict(), 'NCFModel.pth')

if __name__ == '__main__':
    menuData, userData = loadData()

    #Initialize CNN Model
    CNNModel = CNNModel(dimensions = 100, numFilters = 50, filterSize = [3,4,5])
    CNNModel.load_state_dict(torch.load('cnn_model.pth'))
    CNNModel.eval() #Set to evaluation, not training mode
    
    #Get the description tensor and send to CNN Model
    description_tensor = convertDescriptions()
    cnnFeatures = CNNModel(description_tensor) 

    category_tensor = convertCategories()

    #Map ItemIDs to CNN features
    itemIdToFeature = dict(zip(menuData['Item_Id'], cnnFeatures.detach().numpy()))
    userData['CNN_Features'] = userData['Item_ID'].map(itemIdToFeature)
    cnnFeatures = torch.tensor(userData['CNN_Features'].tolist(), dtype=torch.float32)


    #Train NCF Model
    



