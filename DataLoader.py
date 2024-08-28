import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch


def loadData():
    menu_data = pd.read_csv('/Users/shirleyhuang/Documents/Apps/RecommendIQ/Menu.csv')
    user_data = pd.read_csv('/Users/shirleyhuang/Documents/Apps/RecommendIQ/User_Data.csv')

    
    return menu_data, user_data


#Raw Text into numerical vectors (tensor) for CNN then converted to TF-IDF embeddings
def convertDescriptions(menu_data):
    descriptions = menu_data['Description'].tolist()
    vectorizer = TfidfVectorizer()
    descriptions_embeddings = vectorizer.fit_transform(descriptions).toarray()
    tensor = torch.tensor(descriptions_embeddings, dtype = torch.float32)
    print(f"DESCRIPTION TENSOR: {tensor}\n\n")
    return tensor



# Converts categories into one-hot encoded features
def convertCategories(menu_data):
    item_categories = menu_data[['Item_ID', 'Category']].copy()
    item_categories = pd.get_dummies(item_categories, columns=['Category'])

    #Create dictionary mapping IDs to category then assign to userData
    categoriesID = dict(zip(menu_data['Item_ID'], item_categories.drop(columns=['Item_ID']).values))

    print(f"CATEGORY ID: {categoriesID}\n\n")
    return categoriesID


