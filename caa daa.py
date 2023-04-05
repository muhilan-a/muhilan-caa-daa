#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/SG-Akshay10/Dynamic_Programming/blob/main/GeneticAlgorithm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random


# In[53]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.utils import to_categorical


# In[14]:


from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


# In[15]:


import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms


# In[16]:


data = pd.read_csv(r"C:\Users\anton\Downloads\Bank_Personal_Loan_Modelling (2).csv")


# Columns of the dataset : 
# * ID: Customer ID
# * Age: Customer Age
# * Experience: Amount of work experience in years
# * Income: Amount of annual income (in thousands)
# * Zipcode: Zipcode of where customer lives
# * Family: Number of family members
# * CCAvg: Average monthly credit card spendings
# * Education: Education level (1: Bachelor, 2: Master, 3: Advanced Degree)
# * Mortgage: Mortgage of house (in thousands)
# * Securities Account: Boolean of whether customer has a securities account
# * CD Account: Boolean of whether customer has Certificate of Deposit account
# * Online: Boolean of whether customer uses online banking
# * CreditCard: Does the customer use credit card issued by the bank?
# * Personal Loan: This is the target variable (Binary Classification Problem)

# In[17]:


# We can drop the column Customer ID as they do not help us in the prediction.
df = data.drop(columns=["ID"],axis=1)
df


# ## Exploratory Data Analysis

# In[18]:


df.info()


# In[19]:


df.describe().transpose()


# In[20]:


df.isna().any()


# In[21]:


# Percentage of customers having credit cards

CC_percent=(len(df[df['CreditCard']==1])/len(df))*100

print('The percentage of customers having credit cards is', CC_percent, '%')


# In[22]:


#Number of customers who accepted a loan
accepted_customers= df[df['Personal Loan']==1]

# Percentage of customers who accepted a load
accepted_percent=(len(accepted_customers)/len(df))*100

print('The percentage of customers who accepted a loan', accepted_percent, '%')


# ## Data Visualization

# In[23]:


df['Education'].value_counts()


# In[24]:


# Visualize the Personal loan feature
plt.figure(figsize=(5,5))
sns.countplot(data=df, x="Personal Loan")


# In[25]:


# Visualize the education feature
plt.figure(figsize=(5,5))
sns.countplot(data=df, x="Education")


# In[26]:


# Visualize the age feature
plt.figure(figsize=(20,10))
sns.countplot(data=df, x="Age")


# In[27]:


# Visualize credit card availability

plt.figure(figsize=(10,7))
sns.countplot(data=df, x="CreditCard")


# In[28]:


# Visualize income data

sns.distplot(df['Income'])


# In[29]:


personal_loans = df[df['Personal Loan'] == 1].copy()
no_personal_loans=df[df['Personal Loan']==0]
plt.figure(figsize=(15,8))
sns.distplot(personal_loans["Income"], label='Approved')
sns.distplot(no_personal_loans["Income"], label='Not Approved')
plt.legend()
plt.show()


# ## Train Test Split

# In[30]:


df.shape


# In[31]:


x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# In[32]:


x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.25, random_state=69)


# In[33]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[34]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # **# PyTorch Neural Network**

# In[35]:


batch_size = 64


# In[36]:


train_x = torch.from_numpy(x_train).to(torch.float32)
train_y = torch.from_numpy(y_train).to(torch.float32)


# In[37]:


test_x = torch.from_numpy(x_test).to(torch.float32)
test_y = torch.from_numpy(y_test).to(torch.float32)


# In[38]:


class Data(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.x.shape[0]
       
    def __getitem__(self, index):
        return self.x[index], self.y[index]
   
    def __len__(self):
        return self.len


# In[39]:


train_x.shape, train_y.shape


# In[40]:


train_data = TensorDataset(train_x,train_y)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


# In[41]:


test_data = TensorDataset(test_x,test_y)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


# # **# Building Model**

# In[42]:


import torch.nn as nn
import torch.nn.functional as F

class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(12, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


# In[43]:


neural_network = MyNeuralNetwork()


# # **# Weight Optimization using Genetic Algorithm**

# In[44]:


def fitness_function(model):
  y_pred = model(x_train )
  y_pred = torch.where(y_pred>=0.5,1,0).flatten()
  accuracy = (y_pred == train_y).sum().float().item() / len(train_dataloader.dataset)
  return round(accuracy, 4)


# In[45]:


def mutate_child(child, matrix_size):
    # Child Mutation
    random_start = random.randrange(0, matrix_size//2)
    random_end = random.randrange(random_start, matrix_size)
    child_mutate = child.copy()
    child_mutate[random_start:random_end] = child_mutate[random_start:random_end][::-1]

    return child_mutate


# In[46]:


import numpy as np
import torch
import random

def crossover(parent1, parent2, mutation_rate):
    # Extract shapes of weights and biases
    shapes = [param.shape for param in parent1.parameters()]
    
    # Flatten parameters for crossover
    params1 = np.concatenate([param.detach().cpu().numpy().flatten() for param in parent1.parameters()])
    params2 = np.concatenate([param.detach().cpu().numpy().flatten() for param in parent2.parameters()])
    
    # Crossover point
    crossover_point = random.randint(0, len(params1)-1)
    
    # Create children with genes from parents
    child1 = np.concatenate((params1[:crossover_point], params2[crossover_point:]))
    child2 = np.concatenate((params2[:crossover_point], params1[crossover_point:]))
    
    # Mutate children
    child1 = mutate(child1, mutation_rate)
    child2 = mutate(child2, mutation_rate)
    
    # Convert arrays back to tensors
    children = [child1, child2]
    output = []
    
    for child in children:
        params = []
        curr_index = 0
        
        for shape in shapes:
            size = np.prod(shape)
            subset = child[curr_index:curr_index+size]
            array = torch.tensor(subset.reshape(shape), dtype=torch.float32)
            params.append(array)
            curr_index += size
        
        output.append(params)
    
    return output

def mutate(child, mutation_rate):
    # Random mutation on weights and biases
    for i in range(len(child)):
        if random.random() < mutation_rate:
            child[i] += np.random.normal(scale=0.1)
    return child


# In[47]:


# Training 
torch.manual_seed(420)
torch.set_grad_enabled(False)
population_size = 1000


# In[48]:


def train(generations):
  population  = np.array([MyNeuralNetwork() for i in range(population_size)])
  best = None

  for i in range(generations):
    population = sorted(population, key=lambda x: fitness_function(x))
    best = population[-1]
    if(i%10)==0:
      print(f"Generation {i} : {(population[-1])}")

    parents = population[-4:]
    parent_1, parent_2 = population[:2]

    outputs = [crossover(parents[i], parents[i+2]) for i in range(2)]
    output = np.concatenate(outputs)

    new_population = np.array([MyNeuralNetwork() for i in range(len(output))])
    for i, model in enumerate(new_population,0):
      for j, param in enumerate(model.parameters(),0):
        param.data = (torch.tensor(output[i][j]))

    new_population = np.concatenate([new_population, [parent_1, parent_2]])
    population = new_population.copy()
  
  return best


# In[49]:


best_model = train(100)


# # **Classification Report**

# In[52]:


y_pred = best_model(test_x)
y_pred = torch.where(y_pred>=0.5, 1, 0).flatten()
genetic = classification_report(y_pred,test_y)
print(genetic)


# In[ ]:




