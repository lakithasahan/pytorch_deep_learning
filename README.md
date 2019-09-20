# pytorch_deep_learning
In this example we use Pytorch deep learning library to classify  famous IRIS dataset.You can download the complete code with dataset from download section.

## Below shows a simple structure of a Neural Network 

![2 -ann-structure](https://user-images.githubusercontent.com/24733068/65293767-b2597180-db9f-11e9-8293-f7b5c78c7b1b.jpg)

## Lets see how we can implement a simple 2 layer Neural Network using python library numpy.

## As you can observe from the above nn structure we need to define few levels(Input,Hidden and Output).

**Step 1**- You need to make a Forward pass,Inputs should be multiply elements wise with the initaly initialised random weights w1 at        the 1st layer then output from that layer should passed through a activation function such as RELU to the next layer to multiply with the next layer with predefined weights w2 to obtain the initial prediction output.  

**Step 2**- Now you need find the Loss of data by compairing predicted output in step 1 with the desired output it should deliver.

**Step 3**- In this step we need to compute the gradients of w1 and w2 with respect to loss calculated in above step. 

**Step 4**- Now you need use that gradients of w1 and w2(w1_grad,w2_grad)to update the existing weights w1 and w2 with a predefined learning rate.                                                                                                                                  


**Repeat the above steps several time to obtain desired result(epochs)**


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

## At its core, PyTorch provides two main features:

   **An n-dimentional Tesnor,Similar to numpy but with GPU accelaration**                                                                             
   **Automatic diffrentiation for building and training neural networks**                                                                           
   

**Below shows a PyTorch cheat sheet which will be extreamly important in creating your desired network**

![pytorch-cheat](https://user-images.githubusercontent.com/24733068/65290717-b5e6fb80-db93-11e9-905f-159e41df2f30.jpg)

## Below show a example code i wrote to classify IRIS data using PyTorch deep learning.Complete code can be found in the download section.
In PyTorch all your layers can be stacked in nn.sequential package given, For Further undestanding NN i created to classify IRIS data consit Input level,one Hidden level and a Output level.

Number of attributes in the IRIS dataset=4                                                                                                              
Number of outputs in IRIS dataset=3                                                                                                                     
Assumed Number of Hidden Layers=100                                                                                                                                     
```python

model = nn.Sequential(

    nn.Linear(n_in, n_h),   # Initialy data send to Linear(Input) layer (4X100)
    torch.nn.ReLU(),        # and then the result above layer goes to Activation function  
    nn.Linear(n_h, n_h),    #Then result output from the above activation layers goes to next Linear(Hidden) Layer(100x100) 
    torch.nn.ReLU(),        #Result from Hidden layer goes to a activation function
    nn.Linear(n_h, n_out),  #Then result output from the above activation layers goes to next Linear(output) Layer(100x3) 
  

)

```
```python

#define loss function 
criterion = nn.CrossEntropyLoss()

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(tensor_train_x)

    # Compute and print loss
    loss = criterion(y_pred, tensor_train_y)
    if epoch % 100 == 0:
        print('number of epoch', epoch, 'loss', loss.item())
        
        
   # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()


```

## For further in depth undestanding functionalties of PyTorch Please do refer to below tutorial.
https://www.tutorialspoint.com/pytorch/index.htm








