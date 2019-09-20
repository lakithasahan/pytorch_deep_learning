# pytorch_deep_learning
In this example we use Pytorch deep learning library to classify  famous IRIS dataset.

![pytorch-cheat](https://user-images.githubusercontent.com/24733068/65290717-b5e6fb80-db93-11e9-905f-159e41df2f30.jpg)



















```

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
    nn.Linear(n_h, n_h),
    torch.nn.ReLU(),
    nn.Linear(n_h, n_out),
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


```




