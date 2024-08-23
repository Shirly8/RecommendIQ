import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.model_selection import train_test_split

menu_data = pd.read_csv('Menu.csv')
user_data = pd.read_csv('user_data.csv')
num_users = user_data['User_ID'].nunique()
num_items = menu_data['Item_Id'].nunique()

#STEP 1: COLLABORATIVE FILTERING
def collaborativeFiltering():
    sparsity = 1 - ((len(user_data))/(num_users*num_items))
    print(f"Sparsity: {sparsity}")

    #Split dataset
    trainData, testData = train_test_split(user_data)
    #Find cosine similarity ()
if __name__ == '__main__':
    collaborativeFiltering()