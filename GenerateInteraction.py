import csv
import random
import os

def generateInteraction(filename, evalfilename):

    headers = ['User_ID', 'Item_ID', 'Rating']
    users = 200
    menu = 105
    num_evaluations = 50
    current_evaluations = 0

    print(os.getcwd())

    #Open CSV file and write content
    with open(filename, 'w', newline='') as csv_file, open(evalfilename, 'w', newline = '') as evalfile:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        eval_writer = csv.writer(evalfile)
        eval_writer.writerow(headers)
        

        non_zero_item = 0
        total_reviews = users * menu
        
        for customer in range(users):
            reviewed_items=set()

            numReviews = random.randint(50,150)    #Each customer should rate 10 or more item
            numPreference = random.randint(5, numReviews//2)    #Each customer will have preference for 1 item or up to 1/2 the item they review
            preference_bias = random.sample(range(1, menu+1), k= numPreference)    

            
            for _ in range(numReviews):
                randomfood = random.randint(1, menu)

                if randomfood not in reviewed_items:

                     #Rate that item is a part of their preference, rank them higher
                    if randomfood in preference_bias:
                        randomrating = random.randint(4,5)  
                    else:
                       randomrating = random.randint(1,5)
                
                    writer.writerow([customer+1, randomfood, randomrating])
                    reviewed_items.add(randomfood)
                    non_zero_item +=1

            print(f"Customer {customer+1} - {reviewed_items} (PREFERENCES: {preference_bias})")

            #Generate same dataset for evalaution:
            if (current_evaluations < num_evaluations):
                eval_reviews = random.sample(list(reviewed_items), k=len(reviewed_items) // 2)

                for item in eval_reviews:
                    if item in preference_bias:
                        eval_rating = random.randint(4,5)
                    else:
                        eval_rating = random.randint(1,5)
                    eval_writer.writerow([customer+1, item, eval_rating])
                current_evaluations +=1
                
                print(f"EVALUATION {customer+1} - {reviewed_items} (PREFERENCES: {preference_bias})")


        #DATA SPARSITY: Where items have not been reviewed by a customer - missing from dataset
        sparsity = 1 - (non_zero_item/total_reviews)
        print(f"Data Sparsity: {sparsity * 100:.2f}%")





if __name__ == '__main__':
    generateInteraction('/Users/shirleyhuang/Documents/Apps/RecommendIQ/User_Data.csv', '/Users/shirleyhuang/Documents/Apps/RecommendIQ/Eval_Data.csv')



    
                
                  
                  

