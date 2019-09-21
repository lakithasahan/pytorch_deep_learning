import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.autograd import Variable
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
device = torch.device("cpu")
device = torch.device("cuda:0")  # Uncomment this to run on GPU

""""
df = pd.read_csv('iris.csv')
output_names = df['variety']
output_ = []
for i in range(len(output_names)):
    flower_name = str(output_names[i])
    print(type(flower_name))
    print(flower_name)
    if (flower_name == 'Setosa'):
        output_.append([0])
    elif (flower_name == 'Versicolor'):
        output_.append([1])
    else:
        output_.append([2])

print(output_)

input_ = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]

train_X, test_X, train_y, test_y = train_test_split(input_, output_, test_size=0.1)
"""

dataset = pd.read_csv('iris.csv')

# transform species to numerics
dataset.loc[dataset.variety == 'Setosa', 'variety'] = 0
dataset.loc[dataset.variety == 'Versicolor', 'variety'] = 1
dataset.loc[dataset.variety == 'Virginica', 'variety'] = 2

train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:4]].values,
                                                    dataset.variety.values, test_size=0.2)

tensor_train_x = Variable(torch.Tensor(train_X).float())
tensor_train_y = Variable(torch.Tensor(train_y).long())

tensor_test_x = Variable(torch.Tensor(test_X).float())
tensor_test_y = Variable(torch.Tensor(test_y).long())

# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, batch_size = 4, 100, 3, 135

print(tensor_test_x)
# print(ll)

model = nn.Sequential(

    nn.Linear(n_in, n_h),
    torch.nn.ReLU(),
    nn.Linear(n_h, 10),
    #torch.nn.ReLU(),
    nn.Linear(10, n_out),
    #nn.Softmax(dim=1),


)

criterion = nn.CrossEntropyLoss()
# Construct the optimizer (Stochastic Gradient Descent in this case)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(tensor_train_x)

    # Compute and print loss
    loss = criterion(y_pred, tensor_train_y)
    if epoch % 100 == 0:
        print('number of epoch', epoch, 'loss', loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


test_pred=model(tensor_test_x)


print(test_pred)
_, predict_y = torch.max(test_pred, 1)

print(tensor_test_y)
print(predict_y)

# print(predict_y.tolist())
# print([train_y])
print('Accuracy', accuracy_score(tensor_test_y, predict_y))
print('micro precision', precision_score(tensor_test_y, predict_y, average='micro'))
print('macro recall', recall_score(tensor_test_y, predict_y, average='macro'))
print('micro recall', recall_score(tensor_test_y, predict_y, average='micro'))


colormap = np.array(['#f50000', '#f58800', '#0b599f'])
pop_a = mpatches.Patch(color='#f50000', label='Setosa')
pop_b = mpatches.Patch(color='#f58800', label='Versicolor')
pop_c = mpatches.Patch(color='#0b599f', label='Virginica')

plt.subplot(2, 1, 1)
plt.scatter(test_X[:, 0], test_X[:, 1], c=colormap[predict_y], cmap='Paired')
plt.title("PyTorch NN Model predicted target output")

plt.subplot(2, 1, 2)
plt.scatter(test_X[:, 0], test_X[:, 1], c=colormap[tensor_test_y], cmap='Paired')
plt.title("Actual target output")


plt.tight_layout()
plt.legend(handles=[pop_a, pop_b, pop_c])
plt.show()
