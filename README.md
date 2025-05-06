# My First Shallow Neural Network with GPU Support (NumPy + CuPy)

This implementation represents a two-layer neural network (one hidden layer and one output layer) built for **binary classification** tasks. The main feature is the ability to perform the calculations on both the **CPU** (using NumPy) and **GPU** (using CuPy, part of the RAPIDS ecosystem), selectable through a parameter at startup.

## Dependencies

* **NumPy:** Fundamental library for numerical computation in Python (CPU).
* **CuPy:** A library compatible with the NumPy API that performs operations on NVIDIA GPUs (requires installation of the CUDA Toolkit and a compatible GPU).

## Theoretical Foundations

The construction of this algorithm is based on concepts from several areas:

* **Probability:** To understand the sigmoid output as probability and the cost function (cross-entropy).
* **Linear Algebra:** Essential for operations with matrices and vectors (weights, biases, activations, matrix multiplications).
* **Calculus:** Essential for the backpropagation process, which uses partial derivatives (gradients) to adjust the weights.
* **Programming Logic:** To structure the algorithm, the training cycle and the auxiliary functions.
* **Biology (Inspiration):** The structure of neurons and layers is inspired, in a simplified way, by the architecture of the brain.

## Functionalities

The `model_2NN` model is capable of:

1. **Training** on binary classification data (with labels 0 or 1).
2. Performing **probabilistic predictions** (returning the probability of an instance belonging to class 1).
3. Performing **class predictions** (returning the predicted label 0 or 1 based on a threshold).

## Algorithm Structure

The model learning process follows the fundamental steps of a neural network:

### 1. Architecture Definition and Initialization

* **Layer Sizes:** The network has:
* An **input layer** (`n_x`) with a size equal to the number of features (attributes) of the data.
* A **hidden layer** (`n_h`) with a configurable number of neurons (hyperparameter `hidden_layer_units`). The activation function used in this layer is the **Hyperbolic Tangent (tanh)**.
* An **output layer** (`n_y`) with **a single neuron** (since it is binary classification). The activation function is the **Sigmoid**, which maps the output to the interval (0, 1), interpreted as the probability of class 1.
* **Parameter Initialization:**
* **Weights (W):** The weight matrices (`W1` and `W2`) are initialized with small random values, usually from a normal distribution, and multiplied by a small factor (e.g. 0.01) to avoid initial saturation of the activations.
* `W1`: Shape `(n_h, n_x)` - Connects the input to the hidden layer.
* `W2`: Shape `(n_y, n_h)` (or `(1, n_h)`) - Connects the hidden layer to the output.
* **Biases (b):** The bias vectors (`b1` and `b2`) are initialized with zeros.
* `b1`: Shape `(n_h, 1)`
* `b2`: Shape `(n_y, 1)` (or `(1, 1)`)

### 2. Training Loop (Iterations)

The heart of the learning is in a loop that iterates a defined number of times (`iterations`). Within each iteration:

* **a) Forward Propagation:**
1. Calculate the activation of the hidden layer:
* `Z1 = W1 . X + b1` (Linear combination of the input `X`)
* `A1 = tanh(Z1)` (Application of the activation function tanh)
2. Calculate the activation of the output layer:
* `Z2 = W2 . A1 + b2` (Linear combination of hidden activations `A1`)
* `A2 = sigmoid(Z2)` (Application of the sigmoid activation function)
3. `A2` represents the **probabilities predicted** by the model with the current epochs.
4. The intermediate values ​​`Z1`, `A1`, `Z2`, `A2` are stored in a `cache` for use in the next step.

* **b) Cost Calculation:**
1. The "distance" between the variation (`A2`) and the true labels (`Y`) is quantified using the **Binary Cross Entropy (Log Loss)** cost function.
2. This cost measures how well the model is performing with the current parameters. The goal is to minimize this value.

* **c) Backward Propagation:**
1. Calculate the gradient of the cost function with respect to the network output (`dZ2 = A2 - Y`). This is essentially the prediction error.
2. Using the design chain rule and the values ​​from the `cache`, calculate the gradients of the cost function with respect to each parameter (`dW2`, `db2`, `dW1`, `db1`).
3. These gradients indicate the direction and magnitude of the adjustment needed in each parameter to *decrease* the cost. The set of gradients is stored in `grads`.

* **d) Parameter Update:**
1. The parameters are adjusted in the opposite direction to the gradient, using the Gradient Descent rule:
`parameter = parameter - learning_rate * parameter_gradient`
2. The `learning_rate` is a crucial hyperparameter that controls the size of the "step" taken towards the minimum of the cost function.
* High values ​​can make the model "skip" the minimum.
* Low values ​​can make convergence very slow.
3. This step updates `W1`, `b1`, `W2`, `b2` for the next iteration.

This cycle (Forward -> Cost -> Backward -> Update) is repeated, and ideally, the cost decreases over the iterations, indicating that the model is learning to map the inputs `X` to the outputs `Y`.

### 3. Prediction (Predict)

After training, the final parameters (those from the last iteration) are stored. To make predictions on new data `X`:

1. Run **Forward Propagation** using the learned parameters.

2. The output `A2` is the **probabilities** (`predict_proba`).

3. To obtain the **class** (0 or 1), a **threshold** is applied, usually 0.5:

* If the probability (`A2`) > 0.5, the prediction is 1.

* Otherwise, the prediction is 0 (`predict`).

* The threshold can be adjusted to optimize specific metrics such as Precision or Recall.

## Vectorization

The code uses vectorized NumPy/CuPy operations (such as `xp.dot`, `xp.sum`, element-wise operations `+`, `*`, `xp.tanh`, `xp.log`, `xp.exp`). This allows processing all training examples (`m` samples) simultaneously, making the computation much more efficient than using explicit Python loops over each example. The only explicit loop needed is the main loop over the `iterations`.
