# Arabic Name Generator Using Bigram Model

This project implements a minimal character-level bigram language model using PyTorch to generate Arabic names.

## Features
- One-hot encoded bigram training
- Simple forward pass using matrix multiplication
- Softmax-based text generation
- Negative log-likelihood loss evaluation

## Dataset
Arabic names (provided in `names.txt`) — you can replace with your own.

## Training Example
```python
loss.backward()
W.data += -50 * W.grad
