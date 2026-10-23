# Homework 6: PyTorch

| File | Question |
|---|---|
| `Q1.ipynb` | Q1: learnable mixture of activation functions (LeNet on Fashion-MNIST) |
| `Q2.ipynb` | Q2: time-series forecasting with a custom `Dataset` (Jena Climate) |
| `Q3.ipynb` | Q3: training the Lecture 15 GPT with the Muon optimizer |
| `model.py` | the GPT model from Lecture 15, used by Q3 |
| `muon.py` | the Muon optimizer, used by Q3 |
| `data/jena_normalized.npz` | the Jena Climate data, split by time and normalized, used by Q2 |
| `data/shakespeare.txt` | the Lecture 15 text dataset, used by Q3 |
| `agent-solutions/` | the coding agent's solution for Q1-4 |
| `.agents/skills/pytorch/` | the `/pytorch` skill for the coding agent in Q1-4 |

Fashion-MNIST (Q1) is downloaded by the notebook into `data/`.

Requirements: PyTorch, torchvision, numpy, pandas, matplotlib.
