import csv
import random
import os

def generateInteraction(filename):

    headers = ['User_ID', 'Item_ID', 'Rating']
    users = 200
    menu = 105

    print(os.getcwd())


    #Open CSV file and write content
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        
        #For each 100 customers, they must give review to all the menus (Making data sparse)
        for customer in range(users):
            randomNum= 150
            reviewed_items=set()
            
            for _ in range(randomNum):
                randomfood = random.randint(1,menu)

                if randomfood not in reviewed_items:
                    randomrating = random.randint(1,5)
                    writer.writerow([customer+1, randomfood, randomrating])
                    reviewed_items.add(randomfood)

            print(f"Customer {customer+1} - {reviewed_items}")

if __name__ == '__main__':
    generateInteraction('/Users/shirleyhuang/Documents/Apps/RecommendIQ/User_Data.csv')

    
                
                  
                  

