import torch
from DataLoader import loadData, convertCategories, convertDescriptions
from LSTM_Model import train_LSTM
from NCF_Model import train_NCF
import numpy as np


def main():
    menuData, userData = loadData()

    #Get the description and category tensor and send to LSTM Model
    description_tensor = convertDescriptions(menuData)
    lstm_model, description_features = train_LSTM(description_tensor)
    torch.save(lstm_model.state_dict(), 'lstm_model.pth')


    #Map ItemIDs to LSTM features
    itemIdToFeature = dict(zip(menuData['Item_ID'], description_features.detach().numpy()))
    userData['LSTM_Features'] = userData['Item_ID'].map(itemIdToFeature)
    LSTM_features = torch.tensor(np.array(userData['LSTM_Features'].tolist()), dtype=torch.float32)


    # Map Item_IDs to Category features for each user interaction
    categoriesID = convertCategories(menuData)
    userData['Category_Features'] = userData['Item_ID'].map(categoriesID)
    category_tensor = torch.tensor(np.array(userData['Category_Features'].tolist()), dtype=torch.float32)

    #Train NCF Model
    train_NCF(userData, LSTM_features, category_tensor)

if __name__ == '__main__':
    main()