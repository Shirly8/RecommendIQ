import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

menu_data = pd.read_csv('Menu.csv')
user_data = pd.read_csv('user_data.csv')
num_users = user_data['User_ID'].nunique()
num_items = menu_data['Item_ID'].nunique()

def loadData():
    return menu_data, user_data


#Raw Text into numerical vectors (embeddings) for CNN then converted to TF-IDF embeddings
def convertDescriptions():
    descriptions = menu_data['Description'].tolist()
    vectorizer = TfidfVectorizer()
    descriptions_embeddings = vectorizer.fit_transform(descriptions).toarray()
    return descriptions_embeddings

#Categories: One-hot encoding into binary columns
def convertCategories():
    item_categories = menu_data[['Item_Id', 'Category']].copy()
    item_categories = pd.get_dummies(item_categories, columns=['Category'])
    return item_categories


