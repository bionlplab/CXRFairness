import os
import numpy as np
import pandas as pd


def get_id(data_path,label_path,data_id):
    tmp = pd.read_csv(label_path)
    Path = tmp['Path'].tolist()
    com = tmp['Path'].str.split('/').str[0]
    com1 = com.tolist()
    #sex: 1, race : 2, age:3, label:6
    gender = tmp['Sex'].tolist()
    label = tmp['label'].tolist()
    age = tmp['Age'].tolist()
    race = tmp['Race'].tolist()
    
    images_path = []
    labels = []
    groups = []
    le = 0
    com2 = np.array(com1)
    for i in range(len(data_id)):
        ind = np.argwhere(com2 == data_id[i]).ravel()
#         ind = [k for k, x in enumerate(com1) if x == data_id[i]]
        for j in range(len(ind)):
            data_paths = data_path + '/' + tmp['Path'][ind[j]]
            images_path = np.append(images_path,data_paths)
            
            if label[ind[j]] == 'No':
                labels.append([float(0)])
            else:
                labels.append([float(1)])
                      
#             ##for sex
#             if gender[ind[j]] == 'Male':
#                 # groups = np.append(groups,0)
#                 groups.append(0)
#             else:
#                 # groups = np.append(groups,0)
#                 groups.append(1)
            
#             ##for age
#             if age[ind[j]] < 75:
#                 # groups = np.append(groups,0)
#                 groups.append(0)
#             else:
#                 # groups = np.append(groups,0)
#                 groups.append(1)

                
            ##for race    
            if race[ind[j]] == 'White':
                # groups = np.append(groups,0)
                groups.append(0)
            elif race[ind[j]] == 'Black or African American': 
                groups.append(1)
            else:
                # groups = np.append(groups,0)
                groups.append(2)
                

                    
    return images_path, labels, groups