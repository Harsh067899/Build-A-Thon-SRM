# 5G Network Slicing Simulator with AI-Driven Resource Allocation

This repository contains a 5G network slicing simulator with AI-driven resource allocation capabilities. The simulator is designed to demonstrate how machine learning can be used to optimize network slice resource allocation in 5G networks, compliant with 3GPP standards.

## Features

- Simulation of 5G network slices with different service types (eMBB, URLLC, mMTC)
- AI-driven resource allocation using LSTM and autoregressive models
- Implementation of 3GPP standards for network slicing
- Visualization of network performance metrics
- Support for various traffic patterns and scenarios

## New Feature: Autoregressive LSTM Predictor

We've enhanced our AI capabilities with a new **Autoregressive LSTM Predictor** that provides multi-step forecasting for network slice resource allocation. This feature allows the system to predict optimal slice allocations for multiple future time steps, enabling more proactive resource management.

### Key Enhancements

1. **Multi-step Prediction**: Forecast slice allocations for multiple future time steps
2. **Autoregressive Architecture**: Each prediction step feeds into the next for improved temporal coherence
3. **Scenario-based Training**: Model trained on diverse network scenarios including baseline, emergency, smart city, and dynamic traffic patterns
4. **Enhanced Slice Manager**: Integration with the slice management system for real-time allocation decisions
5. **3GPP Standards Compliance**: Fully aligned with 3GPP TS 23.501 slice/service types and QoS parameters

### Usage

To use the autoregressive LSTM predictor:

```python
from slicesim.ai.enhanced_lstm_predictor import AutoregressiveLSTMPredictor

# Create predictor
predictor = AutoregressiveLSTMPredictor(
    input_dim=11,
    sequence_length=10,
    out_steps=5
)

# Make multi-step predictions
predictions = predictor.predict(input_sequence, return_all_steps=True)
```

To use the enhanced slice manager:

```python
from slicesim.ai.slice_manager import EnhancedSliceManager

# Create slice manager
slice_manager = EnhancedSliceManager(
    input_dim=11,
    sequence_length=10,
    out_steps=5
)

# Get optimal slice allocation
allocation = slice_manager.get_optimal_slice_allocation(current_state)
```

### Demo Script

We've included a demo script to showcase the capabilities of the autoregressive LSTM predictor:

```bash
python -m slicesim.ai.demo_enhanced_lstm --show_plots --demo_slice_manager
```

Options:
- `--input_dim`: Dimension of input features (default: 11)
- `--sequence_length`: Length of input sequence (default: 10)
- `--out_steps`: Number of future steps to predict (default: 5)
- `--num_samples`: Number of training samples to generate (default: 10000)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 64)
- `--output_dir`: Directory to save output files (default: 'output')
- `--retrain`: Retrain the model even if a saved model exists
- `--show_plots`: Show plots during execution
- `--demo_slice_manager`: Demo the enhanced slice manager

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/5G-Network-Slicing.git
cd 5G-Network-Slicing

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- TensorFlow 2.4+
- NumPy
- Matplotlib
- Pandas
- scikit-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- 3GPP for the network slicing standards
- TensorFlow team for the machine learning framework
- Context7 for the autoregressive LSTM implementation insights
