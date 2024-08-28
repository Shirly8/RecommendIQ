import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NCFModel(nn.Module):

    def __init__(self, numUsers, numItems, dimensions, featuresDimensions, categoryDimensions):
        super(NCFModel, self).__init__()

        # Create embeddings for users and items
        self.user_embedding = nn.Embedding(numUsers, dimensions)
        self.item_embedding = nn.Embedding(numItems, dimensions)


        # Category embedding (assuming categorical features are integers)
        self.category_embedding = nn.Linear(categoryDimensions, dimensions)

        # Initialize fc1 with correct input size
        input_dim = dimensions * 3 + featuresDimensions  # Including category dimensions
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)  # Output score

    def forward(self, userID, itemID, features, categories):
        userEmbedding = self.user_embedding(userID)
        itemEmbedding = self.item_embedding(itemID)
        
        # Get category embedding
        categoryEmbedding = torch.relu(self.category_embedding(categories))

        # Concatenate all embeddings and features
        x = torch.cat([userEmbedding, itemEmbedding, features, categoryEmbedding], dim=1)

        # Pass through fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x




def train_ncf(userData, features, categories, batch_size=32):
    numUsers = userData['User_ID'].nunique()
    numItems = userData['Item_ID'].nunique()
    dimensions = 20
    categoryDimensions = categories.shape[1]  # Category dimensions size

    # Initialize the NCF model with category dimensions
    model = NCFModel(numUsers, numItems, dimensions, features.shape[1], categoryDimensions)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    losses = nn.MSELoss()

    # Convert data to tensors
    userID = torch.tensor(userData['User_ID'].values - 1, dtype=torch.long)
    itemID = torch.tensor(userData['Item_ID'].values - 1, dtype=torch.long)
    ratings = torch.tensor(userData['Rating'].values - 1, dtype=torch.float32)
    features = torch.tensor(features, dtype=torch.float32)
    categories = torch.tensor(categories, dtype=torch.float32)  # Ensure categories are passed as tensor

    # Create DataLoader for batching
    dataset = TensorDataset(userID, itemID, features, categories, ratings)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()

    for epoch in range(10):
        for batch_userID, batch_itemID, batch_features, batch_categories, batch_ratings in data_loader:
            optimizer.zero_grad()  # Clears previous gradients
            prediction = model(batch_userID, batch_itemID, batch_features, batch_categories)
            loss = losses(prediction.squeeze(), batch_ratings)
            loss.backward()  # Compute gradients
            optimizer.step()

        print(f'Epoch {epoch + 1}/10, Loss: {loss.item()}')

        torch.save(model.state_dict(), 'NCFModel.pth')
