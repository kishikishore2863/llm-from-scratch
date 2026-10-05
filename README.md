# LLM From Scratch

Hands-on implementations on my path to building a lightweight GPT-style language model from scratch in PyTorch (MCA capstone, PES University).

> Following Andrej Karpathy's *Neural Networks: Zero to Hero* series. Each notebook is re-implemented by me with my own notes and experiments.

## Notebooks

| # | Notebook | What it covers | Colab |
|---|----------|----------------|-------|
| 01 | [Micrograd & Backprop](01-micrograd/01_micrograd_backprop.ipynb) | Derivatives, chain rule, a `Value` autograd engine, Neuron/Layer/MLP, PyTorch cross-check | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fIzKQRj4vIkWTXomRFIzbW6DOsmRD5qP?usp=sharing) |
| 02 | [makemore: Bigram](02-makemore-bigram/02_makemore_bigram.ipynb) | Character bigram LM via counting and via a neural net (softmax + NLL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lxl4HG0N3WtJ4P2Jzt25MIyYUeJE-moP?usp=sharing) |
| 03 | [MLP Character LM](03-mlp-language-model/03_mlp_character_lm.ipynb) | Embeddings, hidden layer, mini-batches, cross-entropy, learning-rate search | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12fTgwcraN7R4wW-5SVHGQNt7Ht2AsYeR?usp=sharing) |

## Results

<!-- TODO: add a loss-curve image and 5 sample generated names, e.g. ![loss](images/mlp_loss.png) -->

## My own additions

<!-- TODO: list what you changed/extended (different dataset, ablation on embedding size / hidden units, etc.) -->

## Next steps

- Wavenet-style hierarchical model, BatchNorm, manual backprop
- **MiniGPT**: decoder-only Transformer with causal multi-head attention, BPE tokenizer, pretraining pipeline, ablations (capstone)

## Run locally

```bash
pip install torch matplotlib numpy jupyter
jupyter notebook
```

Dataset: `data/names.txt` (from [karpathy/makemore](https://github.com/karpathy/makemore)).
