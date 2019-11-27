import numpy as np
np.random.seed(6)
from sklearn.model_selection import ShuffleSplit
#from pybedtools import BedTool
#import pybedtools
import pandas as pd
import tensorflow as tf

# Run the code using GPU
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from keras import backend as K
print("Available GPU on Keras: ",K.tensorflow_backend._get_available_gpus())

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout,BatchNormalization,Activation,Input
from keras.layers import Conv1D, MaxPooling1D,Conv2D,MaxPooling2D,GlobalAveragePooling1D, AveragePooling1D
from keras.layers import Dropout
from keras.optimizers import Adam,RMSprop
from keras import regularizers as kr
from keras import initializers
from keras.callbacks import EarlyStopping
from keras.utils import np_utils, to_categorical
# custom R2-score metrics for keras backend
from tensorflow.python.keras import backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from keras.losses import kullback_leibler_divergence

import os
import sys
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping

from sklearn.metrics import r2_score

apply_sample_weight = False
target_length = 1000
filter_length = 6
ACTIVATION = 'linear'
CALLBACKS = [EarlyStopping(monitor='val_loss', patience=5,mode = 'min')]

def R2_score(y_true, y_pred):
    sys.stdout.wirte("R2 score function")
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot) )

def read_data(bed_file,fasta_file):
    #apply bedtools to read fasta files '/home/h5li/methylation_DMR/data/DMR_coordinates_extended_b500.bed'
    a = pybedtools.example_bedtool( bed_file )
    # '/home/h5li/methylation_DMR/data/mm10.fasta'
    fasta = pybedtools.example_filename( fasta_file )
    a = a.sequence(fi=fasta)
    seq = open(a.seqfn).read()
    #read and extract DNA sequences 
    DNA_seq_list = seq.split('\n')
    DNA_seq_list.pop()
    DNA_seq = []
    m = 10000
    n = 0
    for index in range(len(DNA_seq_list)//2):
        DNA_seq.append(DNA_seq_list[index*2 + 1].upper())
        if len(DNA_seq_list[index*2 + 1]) < m:
            m = len(DNA_seq_list[index*2 + 1])
        if len(DNA_seq_list[index*2 + 1]) > n:
            n = len(DNA_seq_list[index*2 + 1])
    print('The shortest length of DNA sequence is {0}bp'.format(m))
    print('The longest length of DNA sequence is {0}bp'.format(n))
    print('Total Number of input sequence is {0}'.format(len(DNA_seq)))
    return DNA_seq,n,m

def extend_Data(targetLength,dnaSeqList):
    newDNAList = []
    for seq in dnaSeqList:
        if len(seq) < targetLength:
            diff = targetLength - len(seq)
            if diff % 2 == 0:
                seq += 'N' * (diff//2)
                seq = 'N' * (diff//2) + seq
            if diff % 2 ==1:
                seq += 'N' *(diff//2)
                seq = 'N' * (diff//2 + 1) + seq
        newDNAList.append(seq)
    return newDNAList

def chop_Data(targetLength,dnaSeqList):
    #chop DNA sequences to have same length
    Uni_DNA = []
    for s in dnaSeqList:
        if len(s) < targetLength:
            print('Exceptions!')
        diff = len(s) - targetLength
        if diff % 2 == 0:
            side = diff // 2
            Uni_DNA.append(s[side:-side])
        else:
            right = diff // 2
            left = diff// 2 + 1
            Uni_DNA.append(s[left:-right])
    return Uni_DNA

#below are helper methods
def data_aug(seq):
    new_seq = []
    for i in range(len(seq)):
        l = seq[i]
        if l == 'A':
            new_seq.append( 'T' )
        elif l == 'C':
            new_seq.append( 'G' )
        elif l == 'G':
            new_seq.append( 'C' )
        elif l == 'T':
            new_seq.append( 'A' )
        else:
            new_seq.append( 'N' )
    return new_seq

def data_rev(seq):
    new_seq = [None] * len(seq)
    for i in range(len(seq)):
        new_seq[-i] = seq[i]
    return new_seq      

def mse_keras(y_true, y_pred):
    SS_res =  K.sum( K.square( y_true - y_pred ) ) 
    SS_tot = K.sum( K.square( y_true - K.mean( y_true ) ) ) 
    return ( SS_res/SS_tot)

def R2_score(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot) )

def preprocess_data(DNA_seq):

    train_size = len(DNA_seq)

    #One hot encoding 
    DNA = []
    for u in DNA_seq:
        sequence_vector = []
        for c in u:
            if c == 'A':
                sequence_vector.append([1,0,0,0])
            elif c == 'C':
                sequence_vector.append([0,1,0,0])
            elif c == 'G':
                sequence_vector.append([0,0,1,0])
            else:
                sequence_vector.append([0,0,0,1])
        #print(np.array(sequence_vector).shape)
        DNA.append(np.array(sequence_vector))
    DNA = np.array(DNA)
    print(DNA.shape)
    return DNA

def Formalize_Data(DNA_seq, methylation_file, cell_type):
    #Read Methylation level
    labels = list(pd.read_csv(methylation_file,header = None)[cell_type])
    train_labels = np.array(labels)
    training_seq_shape = (len(DNA_seq),len(DNA_seq[0]),4)
    train_data = DNA_seq.reshape(training_seq_shape)
    return train_data,train_labels

bed_file_path =          '/home/h5li/methylation_DMR/data/DMR_coordinates_extended_b500.bed'
fasta_file_path =        '/home/h5li/methylation_DMR/data/mm10.fasta'
methylation_file_path =  '/home/h5li/methylation_DMR/data/Mouse_DMRs_methylation_level.csv'
total_counts_file_path = '/home/h5li/methylation_DMR/data/Mouse_DMRs_counts_total.csv'
methy_counts_file_path = '/home/h5li/methylation_DMR/data/Mouse_DMRs_counts_methylated.csv'
    
#DNA_seq,long_length,short_length = read_data(bed_file_path, fasta_file_path)   


# In[10]:


#DNA_seq = chop_Data(target_length,DNA_seq)


# In[16]:


#DNA = preprocess_data(DNA_seq)
#train_data,train_labels = Formalize_Data(DNA, methylation_file_path, cell_type)
#np.save("DNA_seq.npy",train_data)
#np.save("DNA_methy.npy",train_labels)

from sklearn.model_selection import train_test_split

init = initializers.RandomNormal(mean=0, stddev=0.5, seed=None)

def model_fn(model,num_conv_layers,num_filters,filter_length,maxpool_size,batchnorm,dropout,dense_activation):
    assert num_conv_layers == len(num_filters) == len(filter_length)
    assert num_conv_layers == len(maxpool_size) + 1 
    assert num_conv_layers > 0
    
    for c in range(0,num_conv_layers):
        if c != 0:
            model.add(Conv1D(filters = num_filters[c], 
                             kernel_size=filter_length[c],
                             kernel_initializer = init,
                             padding = 'same',
                             activation = 'relu'))
        
        # Add MaxPooling
        if c == num_conv_layers - 1:
            # If this is the last layer, add global average pooling
            model.add(GlobalAveragePooling1D())
        elif c < len(maxpool_size):
            model.add(MaxPooling1D(maxpool_size[c]))
        
        # Add BatchNormalization
        if batchnorm:
            model.add(BatchNormalization())
    
    if dropout:
        model.add(Dropout(dropout))
    
    model.add(Dense(16, kernel_initializer=init, activation=dense_activation))
    model.compile(optimizer= Adam(lr = 0.005),
                  loss='mean_squared_error')
    
def train_model(data,train_labels,
                num_conv_layers,num_filters,
                filter_length,maxpool_size,
                batchnorm,dropout,dense_activation,seqlen):
    
    model = Sequential()
    model.add(Conv1D(filters=num_filters[0], kernel_size=filter_length[0],kernel_initializer = init,padding = 'same',
                     input_shape=(seqlen,4), activation='relu'))
    model_fn(model,num_conv_layers,num_filters,filter_length,maxpool_size,batchnorm,dropout,dense_activation)
    
    #filename = 'Filters:{0}_FilterSize:{1}_maxpoolSize:{2}_batchnorm:{3}_dropout:{4}%_DenseActivation:{5}'.format(            
    #    num_filters,filter_length,maxpool_size,
    #    batchnorm,int(dropout*100),dense_activation)
    
     
    history = model.fit(data, train_labels, epochs=1000, callbacks = CALLBACKS,
                        validation_split = 0.05,shuffle = True,batch_size=64,verbose=2)
    
    index = history.history['val_R2_score'].index(max(history.history['val_R2_score']))

    return model,[history.history['R2_score'][index],
            history.history['val_R2_score'][index],
            history.history['loss'][index],
            history.history['val_loss'][index]]

def get_model(data,train_labels,
                num_conv_layers,num_filters,
                filter_length,maxpool_size,
                batchnorm,dropout,dense_activation,seqlen):
    
    model = Sequential()
    model.add(Conv1D(filters=num_filters[0], kernel_size=filter_length[0],kernel_initializer = init,padding = 'same',
                     input_shape=(seqlen,4), activation='relu'))
    model_fn(model,num_conv_layers,num_filters,filter_length,maxpool_size,batchnorm,dropout,dense_activation)
    
    #filename = 'Filters:{0}_FilterSize:{1}_maxpoolSize:{2}_batchnorm:{3}_dropout:{4}%_DenseActivation:{5}'.format(            
    #    num_filters,filter_length,maxpool_size,
    #    batchnorm,int(dropout*100),dense_activation)
    return model
    
     

def binarize_level(level):
    processed_level = []
    for i,v in enumerate(level):
        if v >= 0.5:
            processed_level.append(1)
        else:
            processed_level.append(0)
    return np.array(processed_level)

rs = ShuffleSplit(n_splits=5, test_size=0.2, random_state=24)

train_data = np.load("../DNA_seq.npy")[:,200:800,:]
train_methys = pd.read_csv('../../data/Mouse_DMRs_methylation_level.csv',header = None)

print(train_data.shape,train_methys.shape)

batch_size = 64
validation_split = 0.2
shuffle_dataset = True
random_seed= 24

dataset_size = len(train_data)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

cell_type = 5
X = train_data
train_X = X[train_indices]
val_X = X[val_indices]

multiclass = True
if multiclass:
    Y = np.array(train_methys)
else:
    Y = np.array(train_methys[cell_type])

train_Y = Y[train_indices]
val_Y = Y[val_indices]

num_layers = 2
filters = 32
filt_size = 3
dropout = 0.25
activation = 'linear'

maxpool  = 3
second_num_filters = 8
second_filter_size = 6

seqlen = 600        

model = get_model(train_X,train_Y,
                            num_layers,
                            [filters,second_num_filters],
                            [filt_size,second_filter_size],
                            [maxpool],True,dropout,activation,seqlen)

for e in range(100):
    model.fit(train_X, train_Y, epochs=1,
              shuffle = True,batch_size=64,verbose=2)
    model.evaluate(val_X,val_Y,verbose = 2)
    pred = model.predict(val_X)
    score_num = 0 
    for i in range(16):
        sys.stdout.write("T:{} S:{:.4f} ".format(i,r2_score(val_Y[:,i],pred[:,i])))
        score_num += r2_score(val_Y[:,i],pred[:,i])
    sys.stdout.write(" SUM:{}".format(score_num))
    sys.stdout.write("\n")
