import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd



def remove_escape_char(array):
    for ind,elem in enumerate(array):
        array[ind]=elem.replace('\n','').replace('(','').replace(')','').replace(',','')
    return array

def convert_to_numpy(array):
    train_data=[]
    for ind,elem in enumerate(array): 
        data=elem.split(' ')
        data=np.array(list(map(int,data)))
        train_data.append(data)
    return np.array(train_data)

def file_read(txt_file): 
    with open(txt_file) as fd:
        file=fd.readlines()
        file=remove_escape_char(file)
    return convert_to_numpy(file)





if __name__=='__main__':
    blue=file_read('blue.txt')
    blue_label=["blue"]*1200

    green=file_read('green.txt')
    green_label=["green"]*1200

    orange=file_read('orange.txt')
    orange_label=["orange"]*1200

    red=file_read('red.txt')
    red_label=["red"]*1200

    white=file_read('white.txt')
    white_label=["white"]*1200

    yellow=file_read('yellow.txt')
    yellow_label=["yellow"]*1200

    data=np.concatenate((blue,green,orange,red,white,yellow))
    label=blue_label+green_label+orange_label+red_label+white_label+yellow_label
    
    dictionary={"r":data[:,0],"g":data[:,1],"b":data[:,2],"label":label}
    df = pd.DataFrame(dictionary)
    df.to_csv('file1.csv',header=True,index=False)

    
    
