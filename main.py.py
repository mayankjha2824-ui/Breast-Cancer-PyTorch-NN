import numpy as np 
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv')
print(df.head())
df.drop(columns = ['id','Unnamed: 32'], inplace = True)
print(df.head())
X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,1:],df.iloc[:,0],test_size = 0.2)
print(X_train)
print(y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
print(y_train)
X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train).float()
y_test_tensor = torch.from_numpy(y_test).float()
import torch.nn as nn 
class Model(nn.Module):
   def __init__(self,num_features):
      super().__init__()
      self.network = nn.Sequential(nn.Linear(num_features,3),nn.ReLU(),nn.Linear(3,1),nn.Sigmoid())
   def forward(self,features):
      out = self.network(features)
      return out 
learning_rate = 0.1
epochs = 25
loss_function = nn.BCELoss()
model = Model(X_train_tensor.shape[1])
optimizer = torch.optim.Adam(model.parameters(),lr= learning_rate)
for epoch in range(epochs):
   y_pred = model.forward(X_train_tensor)
   loss = loss_function(y_pred,y_train_tensor.view(-1,1))
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")
with torch.no_grad():
   y_pred = model.forward(X_test_tensor)
   y_pred = (y_pred > 0.5).float()
   accuracy = (y_pred==y_test_tensor).float().mean()
   print(f"Accuray: {accuracy.item()*100:.2f}")

   


  
