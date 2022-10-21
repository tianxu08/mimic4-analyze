import os
import csv
import random
p = '/Users/xutian/Documents/Dev/dartmouth/research/mimic4-analyze/data/root'


if __name__ == "__main__":
    print("aaa")
    ids = [name for name in os.listdir(p) if os.path.isdir(p)]
    train_ids = []
    test_ids = []
    with open('testset.csv', 'w') as f:
        for id in ids: 
            isTest = random.randint(0, 100) % 5 == 0
            print("isTest: ", isTest)
            if isTest:
                line = id + ',' + str(1)
            else:
                line = id + ',' + str(0)
                
            print(line)
            f.write(line+'\n')
                
        
        
            
                
        
    
    

