# PyTorch Fundamentals – Neural Network Implementation

This project documents my hands-on practice with **PyTorch**, focusing on understanding tensors, automatic differentiation, model building, and training a neural network using PyTorch’s core APIs.

---

## Project File

- Pytorch.py – Python script containing the complete PyTorch workflow

---

## 1. Objective

The objective of this work is to:
- Learn PyTorch fundamentals through hands-on coding
- Understand how tensors and gradients work
- Build and train a neural network using PyTorch
- Implement a manual training loop instead of using high-level abstractions

---

## 2. Working with PyTorch Tensors

In the initial part of the notebook:
- PyTorch tensors are created and manipulated
- Tensor shapes and operations are explored
- The difference between regular tensors and tensors with `requires_grad=True` is observed

This builds the foundation for understanding how PyTorch handles numerical computation.

---

## 3. Automatic Differentiation (Autograd)

Steps performed:
- Enable gradient tracking on tensors
- Perform mathematical operations
- Use `.backward()` to compute gradients
- Inspect gradients stored in `.grad`

This section demonstrates how PyTorch automatically computes gradients for optimization.

---

## 4. Dataset Preparation

- Input features and target labels are prepared as tensors
- Data is formatted to be compatible with PyTorch models
- Dataset is split logically for training purposes

This step prepares raw data for model training.

---

## 5. Building the Neural Network Model

A neural network is defined using:
- `torch.nn.Module`
- Linear layers (`nn.Linear`)
- Activation functions

The model structure is explicitly defined to understand how layers connect and pass data forward.

---

## 6. Defining Loss Function and Optimizer

- A suitable loss function is chosen for the task
- An optimizer is defined to update model parameters
- Model parameters are passed to the optimizer

This connects the model with the learning objective.

---

## 7. Training Loop Implementation

A **manual training loop** is implemented:
- Forward pass through the model
- Loss computation
- Backward pass using `loss.backward()`
- Parameter updates using `optimizer.step()`
- Gradients are reset using `optimizer.zero_grad()`

This section focuses on understanding what happens internally during training.

---

## 8. Model Evaluation

- Model predictions are generated after training
- Output values are inspected
- Model behavior is analyzed based on predictions

This confirms whether the model learned from the data.

---

## 9. Key Observations

- PyTorch provides fine-grained control over training
- Gradient computation is handled automatically via autograd
- Manual training loops improve understanding of deep learning internals
- Clearing gradients after each iteration is critical

---

## 10. Conclusion

This work serves as a foundational introduction to PyTorch. By manually building tensors, defining models, and writing a training loop, I gained a clear understanding of how deep learning models are trained at a low level using PyTorch.

---

