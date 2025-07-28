# Updated with gradient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# load the dataset
data = pd.read_csv('backend/dataset/train_dataset.csv')
X = data['feature_0'].values
Y= data['target'].values

m = 0
b = 0
lr = 0.01
epochs = 1000
n = len(X)

for epoch in range(epochs):
    y_pred = m * X + b
    error = y_pred - Y
    
    #grediants
    dm = (2/n) * np.dot(error, X)
    db = (2/n) * np.sum(error)
    
    #update weights
    m -= lr * dm
    b -= lr * db
    
    if epoch % 100 == 0:
        mse = np.mean(error ** 2)
        print(f"Epoch {epoch}: MSE= {mse:.4f}, m = {m:.4f}, b = {b:.4f}")




# OLD Implementation
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt 

# # load the dataset
# data = pd.read_csv('backend/dataset/train_dataset.csv')
# X = data['feature_0'].values
# Y= data['target'].values

# #Mean
# x_mean = np.mean(X)
# y_mean = np.mean(Y)

# # slope and intercept
# m = np.sum((X - x_mean) * (Y - y_mean)) / np.sum((X - x_mean) ** 2)
# b = y_mean - m * x_mean

# def predict(x):
#     return m * x + b

# y_pred = predict(X)


