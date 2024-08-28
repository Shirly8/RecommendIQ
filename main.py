import torch
from DataLoader import loadData, convertCategories, convertDescriptions
from CNN_Model import train_cnn
from NCF_Model import train_ncf
import numpy as np


def main():
    menuData, userData = loadData()

    #Get the description and category tensor and send to CNN Model
    description_tensor = convertDescriptions(menuData)
    categoriesID = convertCategories(menuData)

    # Train the CNN model on descriptions
    cnn_model, cnnFeatures = train_cnn(description_tensor)
    torch.save(cnn_model.state_dict(), 'cnn_model.pth')


    #Map ItemIDs to CNN features
    itemIdToFeature = dict(zip(menuData['Item_ID'], cnnFeatures.detach().numpy()))
    userData['CNN_Features'] = userData['Item_ID'].map(itemIdToFeature)
    cnnFeatures = torch.tensor(np.array(userData['CNN_Features'].tolist()), dtype=torch.float32)



    # Map Item_IDs to Category features for each user interaction
    userData['Category_Features'] = userData['Item_ID'].map(categoriesID)
    category_tensor = torch.tensor(userData['Category_Features'].tolist(), dtype=torch.float32)

    #Train NCF Model
    train_ncf(userData, cnnFeatures, category_tensor)

if __name__ == '__main__':
    main()