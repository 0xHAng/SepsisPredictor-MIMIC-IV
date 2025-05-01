# Sepsis Prediction Model

This is a Transformer-based model for predicting the risk of shock and death in sepsis patients.

## Project Structure

```
.
├── README.md
├── config.py              # Configuration file
├── model.py              # Model definition
├── parameter_search.py   # Parameter search
├── utils.py              # Utility functions
├── parameter_search_test_results.csv  # Parameter search results
└── complete_data_and_labels/  # Dataset directory
```

## Requirements

- Python 3.x
- PyTorch
- CUDA (if using GPU)

## Main Features

- Time series data modeling using Transformer architecture
- Support for multi-task learning (shock prediction and death prediction)
- Configurable look-back and prediction windows
- Automatic parameter search functionality

## Model Architecture

The model is based on Transformer architecture and includes the following components:
- Full Attention mechanism (FullAttention)
- Attention Layer (AttentionLayer)
- Encoder Layer (EncoderLayer)
- Encoder (Encoder)
- Embedding Layer (Embed)

## Configuration Parameters

Main configuration parameters include:
- Data Parameters:
  - Look-back window range: [12, 24]
  - Shock prediction window range: [6, 12]
  - Death prediction window range: [168, 336]
  - Minimum records ratio: 0.5
  - ABPm null ratio threshold: 0.2

- Model Parameters:
  - Model dimension ratio: 4
  - Number of attention heads: 8
  - Feed-forward network dimension ratio: 8
  - Number of encoder layers: 2
  - Dropout rate: 0.05
  - Activation function: ReLU

- Training Parameters:
  - Batch size: 256
  - Learning rate: 0.0001
  - Early stopping rounds: 6
  - Device: CUDA

## Usage

1. Setup environment:
```bash
pip install -r requirements.txt
```

2. Prepare data:
- Place data files in the `complete_data_and_labels/` directory

3. Run parameter search:
```bash
python parameter_search.py
```

4. Train the model:
```bash
python train.py
```

## Notes

- Ensure sufficient GPU memory for training
- Data preprocessing steps are implemented in `utils.py`
- Model checkpoints are saved in the `checkpoints/` directory
- Experimental results are saved in the `results/` directory

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

The MIT License allows:
- Commercial use
- Modification
- Distribution
- Private use

With the following conditions:
- Include the original license and copyright notice
- State changes made to the code

Copyright (c) 2024 Hangyu Yuan, University of Electronic Science and Technology of China 