# pytorch_deep_learning
In this example we use Pytorch deep learning library to classify  famous IRIS dataset.

## Below shows a simple structure of a Neural Network 

![2 -ann-structure](https://user-images.githubusercontent.com/24733068/65293767-b2597180-db9f-11e9-8293-f7b5c78c7b1b.jpg)

## Lets see how we can implement a simple 2 layer Neural Network using python library numpy.

## As you can observe from the above nn structure we need to define few levels(Input,Hidden and Output).

**Step 1**- You need to make a Forward pass,Inputs should be multiply elements wise with the initaly initialised random weights w1 at        the 1st layer then output from that layer should passed through a activation function such as RELU to the next layer to multiply with the next layer with predefined weights w2 to obtain the initial prediction output.  

**Step 2**- Now you need find the Loss of data by compairing predicted output in step 1 with the desired output it should deliver.

**Step 3**- In this step we need to compute the gradients of w1 and w2 with respect to loss calculated in above step. 

**Step 4**- Now you need use that gradients of w1 and w2(w1_grad,w2_grad)to update the existing weights w1 and w2 with a predefined learning rate.                                                                                                                                  


**Repeat the above steps several time to obtain desired result(epochs)**

## At its core, PyTorch provides two main features:

   **An n-dimentional Tesnor,Similar to numpy but with GPU accelaration**
   **Automatic diffrentiation for building and training neural networks**
   


### Please find code for simple NN example created using Numpy below.

```python
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 5, 10, 5, 2

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)


    print(h)
    print(h_relu)
    print(ll)
    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

```




## Since now you have a basic idea how the NN can be implement using numpy, lets move to pytorch implementation.

**Below shows a PyTorch cheat sheet which will be extreamly important in creating your desired network**

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




