# pytorch_deep_learning
In this example we use Pytorch deep learning library to classify  famous IRIS dataset.

## Below shows a simple structure of a Neural Network 

![2 -ann-structure](https://user-images.githubusercontent.com/24733068/65293767-b2597180-db9f-11e9-8293-f7b5c78c7b1b.jpg)

## Lets see how we can implement a simple 2 layer Neural Network using python library numpy.

## As you can observe from the above nn structure we need to define few levels(Input,Hidden and Output).

**Step 1**- You need to make a Forward pass,Inputs should be multiply elements wise with the initaly initialised random weights w1 at        the 1st layer then output from that layer should passed through a activation function such as RELU to the next layer to multiply with the next layer random weights w2 to obtain the initial prediction output.  

**Step 2**- Now you need find the Loss of data by compairing predicted output in step 1 with the desired output it should deliver.

**Step 3**- In this step we need to compute the gradients of w1 and w2 with respect to loss calculated in above step. 

**Step 4**- Now you need use that gradients of w1 and w2(w1_grad,w2_grad)to update the existing weights w1 and w2 with a predefined learning rate.                                                                                                                                  





![pytorch-cheat](https://user-images.githubusercontent.com/24733068/65290717-b5e6fb80-db93-11e9-905f-159e41df2f30.jpg)




## At its core, PyTorch provides two main features:

   **An n-dimentional Tesnor,Similar to numpy but with GPU accelaration**
   **Automatic diffrentiation for building and training neural networks**
   















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




