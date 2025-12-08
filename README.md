# Progressive Neural Network (PNN) for Continual Learning

**Author:** Nebiyu Tefera  
**Language:** Python / PyTorch  
**Framework:** PyTorch  
**Purpose:** Implementation of Progressive Neural Networks (PNN) for continual learning tasks.  

---

## Overview

This repository contains my **implementation of Progressive Neural Networks (PNNs)**, a neural network architecture designed for **continual learning**.  
The project includes **LSTM-based and CNN-based columns**, modular **blocks**, and the ability to **add new columns progressively** while preserving knowledge from previously learned tasks.  

**Key features:**
- Modular design with `ProgBlock`, `ProgColumn`, and `ProgNet` classes.
- LSTM and CNN column generators.
- Support for **lateral connections** between columns for knowledge transfer.
- Training and evaluation pipelines for binary or multi-class classification tasks.
- Fully compatible with **GPU acceleration**.

---

## Table of Contents

- [Progressive Neural Network (PNN) for Continual Learning](#progressive-neural-network-pnn-for-continual-learning)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
    - [1. Initialize a column generator:](#1-initialize-a-column-generator)
    - [2. Add a new column and train:](#2-add-a-new-column-and-train)
    - [3. Forward pass:](#3-forward-pass)
  - [Configuration](#configuration)
  - [Training \& Evaluation](#training--evaluation)
  - [Results Visualization](#results-visualization)
  - [References](#references)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Nebagit/Continual-Learning-with-Progressive-Neural-Networks.git
cd Continual-Learning-with-Progressive-Neural-Networks
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `torch`
- `numpy`
- `matplotlib`
- `scikit-learn` (optional for preprocessing)

---

## Project Structure

```text
progressive-neural-network/
│
├── src/
│   ├── Blocks.py               # Implementation of dense, LSTM, and CNN blocks
│   ├── ProgNet.py              # Core classes: ProgBlock, ProgColumn, ProgNet
│   ├── Column_Generator.py     # LSTM and CNN column generators
│   └── config.py               # Configuration file with hyperparameters
```

---

## Usage

### 1. Initialize a column generator:

```python
from src.Column_Generator import Column_generator_LSTM
from src.ProgNet import ProgNet

column_generator = Column_generator_LSTM(
    input_size=200, 
    hidden_size=128, 
    num_of_classes=2, 
    num_LSTM_layer=1, 
    num_dens_Layer=1
)

pnn_model = ProgNet(colGen=column_generator)
```

### 2. Add a new column and train:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
col_id = pnn_model.addColumn(device=device)

# Example: train your column using data
# optimizer = torch.optim.Adam(pnn_model.parameters(), lr=config.learning_rate)
```

### 3. Forward pass:

```python
output = pnn_model(col_id, input_tensor)
```

---

## Configuration

All hyperparameters can be adjusted in `src/config.py`:

| Parameter               | Description                                  | Default |
|-------------------------|----------------------------------------------|---------|
| flow_size               | Number of flows per batch                     | 100     |
| pkt_size                | Number of packets per flow                    | 200     |
| column_batch_size       | Batch size for each column                    | 128     |
| learning_batch_size     | Training batch size within a column          | 16      |
| learning_rate           | Optimizer learning rate                       | 0.001   |
| epochs                  | Training epochs for each column              | 50      |
| task_epochs             | Task-specific fine-tuning epochs             | 20      |
| test_labels             | Labels for testing                            | Binary/Custom |
| all_labels              | Labels for training                           | Custom  |

---

## Training & Evaluation

1. **Training**: Columns are trained sequentially. Each new column learns **without modifying previously learned columns**.
2. **Evaluation**: The `evaluate_precision` function calculates precision on validation or test datasets.

```python
precision = evaluate_precision(pnn_model, X_val, y_val)
print(f"Validation Precision: {precision:.4f}")
```

3. **Save model weights**:

```python
torch.save(pnn_model.state_dict(), 'binary-PNN-weights.pth')
```

---

## Results Visualization

You can visualize the precision over columns:

```python
import matplotlib.pyplot as plt

plt.plot([i * 10 for i in range(number_of_columns)], precisions, color='b')
plt.suptitle('Precision over columns')
plt.show()
```

---

## References

1. Rusu et al., *Progressive Neural Networks*, 2016.  
2. Zhang et al., *cPNN: Continuous Progressive Neural Networks for Evolving Streaming Time Series*, 2023.  
3. Nguyen et al., *Robust Continual Learning through a Comprehensively Progressive Bayesian Neural Network*, 2022.

---

**⭐ Tips:**  
- Use GPU for faster training (`torch.device("cuda")`).  
- Adjust `num_LSTM_layer` or `num_dens_Layer` to experiment with different architectures.  
- Lateral connections allow **transfer learning** across tasks.  
```