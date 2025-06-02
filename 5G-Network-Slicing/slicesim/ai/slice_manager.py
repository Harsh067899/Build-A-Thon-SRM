#!/usr/bin/env python3
"""
Enhanced Network Slice Manager with Autoregressive LSTM Predictor

This module provides an enhanced slice manager that uses the autoregressive LSTM
predictor to make multi-step predictions for optimal slice resource allocation.

3GPP Standards Compliance:
- Implements slice types according to 3GPP TS 23.501 (SST values)
- Supports QoS parameters defined in 3GPP TS 23.501 Section 5.7
- Aligns with Network Slice Selection Assistance Information (NSSAI) concept
"""

import numpy as np
import os
from datetime import datetime
import logging

# Import the autoregressive LSTM predictor
from slicesim.ai.enhanced_lstm_predictor import AutoregressiveLSTMPredictor
from slicesim.ai.lstm_predictor import SliceAllocationPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base SliceManager class
class SliceManager:
    """Base Network Slice Manager
    
    This class provides the foundation for network slice management
    using the LSTM predictor.
    
    3GPP Compliance:
    - SST=1: eMBB (Enhanced Mobile Broadband)
    - SST=2: URLLC (Ultra Reliable Low Latency Communications)
    - SST=3: mMTC (Massive Machine Type Communications)
    
    Where SST is Slice/Service Type as defined in 3GPP TS 23.501
    """
    
    def __init__(self, input_dim=11, sequence_length=10, model_path=None, skip_training=False):
        """Initialize the slice manager
        
        Args:
            input_dim (int): Dimension of input features
            sequence_length (int): Length of input sequence
            model_path (str): Path to load the model from
            skip_training (bool): Whether to skip training during initialization
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.model_path = model_path
        
        # Initialize the predictor
        logger.info("Initializing LSTM predictor...")
        self.predictor = SliceAllocationPredictor(
            input_dim=input_dim,
            sequence_length=sequence_length,
            model_path=model_path,
            skip_training=skip_training
        )
        
        # Initialize history buffer
        self.history_buffer = []
        
        logger.info("Slice manager initialized successfully")
    
    def update_history_buffer(self, current_state):
        """Update the history buffer with the current state
        
        Args:
            current_state (numpy.ndarray): Current state vector
        """
        # Add current state to history buffer
        self.history_buffer.append(current_state)
        
        # Keep only the last sequence_length states
        if len(self.history_buffer) > self.sequence_length:
            self.history_buffer.pop(0)
    
    def get_optimal_slice_allocation(self, current_state):
        """Get optimal slice allocation based on current state
        
        Args:
            current_state (numpy.ndarray): Current state vector
            
        Returns:
            numpy.ndarray: Optimal slice allocation
        """
        # Update history buffer
        self.update_history_buffer(current_state)
        
        # If we don't have enough history, use default allocation
        if len(self.history_buffer) < self.sequence_length:
            logger.info(f"Not enough history ({len(self.history_buffer)}/{self.sequence_length}), using default allocation")
            # Default allocation: equal distribution
            return np.array([1/3, 1/3, 1/3])
        
        # Convert history buffer to numpy array
        input_sequence = np.array(self.history_buffer)
        
        # Get prediction from LSTM model
        prediction = self.predictor.predict(input_sequence)
        
        # Safe logging that handles numpy arrays correctly
        embb = prediction[0].item() if hasattr(prediction[0], 'item') else float(prediction[0])
        urllc = prediction[1].item() if hasattr(prediction[1], 'item') else float(prediction[1])
        mmtc = prediction[2].item() if hasattr(prediction[2], 'item') else float(prediction[2])
        logger.info(f"Predicted allocation: eMBB={embb:.2f}, URLLC={urllc:.2f}, mMTC={mmtc:.2f}")
        
        return prediction
    
    def train_model(self, training_data=None, epochs=150, batch_size=64, validation_split=0.2):
        """Train the LSTM model
        
        Args:
            training_data (tuple): Training data (X, y)
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Validation split ratio
            
        Returns:
            keras.callbacks.History: Training history
        """
        if training_data is None:
            # Generate synthetic training data
            logger.info("Generating synthetic training data...")
            X_train, y_train, X_val, y_val = self.predictor._generate_training_data(15000)
            training_data = (X_train, y_train)
            validation_data = (X_val, y_val)
        else:
            X, y = training_data
            # Split into training and validation sets
            split_idx = int((1 - validation_split) * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            validation_data = (X_val, y_val)
        
        # Train model
        logger.info("Training LSTM model...")
        history = self.predictor.train(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data
        )
        
        # Log training results
        val_loss = history.history['val_loss'][-1]
        val_mae = history.history['val_mae'][-1]
        logger.info(f"Training completed. Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")
        
        return history
    
    def save_model(self, path=None):
        """Save the LSTM model
        
        Args:
            path (str): Path to save the model
        """
        if path is None:
            path = f"models/lstm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        
        self.predictor.save(path)
        logger.info(f"Model saved to {path}")


class EnhancedSliceManager(SliceManager):
    """Enhanced Network Slice Manager with Autoregressive LSTM Predictor
    
    This class extends the base SliceManager with multi-step prediction capabilities
    using the autoregressive LSTM model from Context7 insights.
    
    3GPP Compliance:
    - SST=1: eMBB (Enhanced Mobile Broadband)
    - SST=2: URLLC (Ultra Reliable Low Latency Communications)
    - SST=3: mMTC (Massive Machine Type Communications)
    
    Where SST is Slice/Service Type as defined in 3GPP TS 23.501
    """
    
    def __init__(self, input_dim=11, sequence_length=10, out_steps=5, model_path=None, 
                 checkpoint_dir="models/checkpoints", skip_training=False):
        """Initialize the enhanced slice manager
        
        Args:
            input_dim (int): Dimension of input features
            sequence_length (int): Length of input sequence
            out_steps (int): Number of future steps to predict
            model_path (str): Path to load the model from
            checkpoint_dir (str): Directory to save model checkpoints
            skip_training (bool): Whether to skip training during initialization
        """
        # Initialize the parent class
        super().__init__(input_dim, sequence_length, model_path, skip_training)
        
        # Override the predictor with our autoregressive model
        self.out_steps = out_steps
        self.checkpoint_dir = checkpoint_dir
        
        # Create directory for checkpoints if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize the autoregressive predictor
        logger.info("Initializing autoregressive LSTM predictor...")
        self.predictor = AutoregressiveLSTMPredictor(
            input_dim=input_dim,
            sequence_length=sequence_length,
            out_steps=out_steps,
            model_path=model_path,
            skip_training=skip_training
        )
        
        # Initialize history buffer for input sequences
        self.history_buffer = []
        
        # Initialize prediction history for visualization and analysis
        self.prediction_history = []
        
        logger.info("Enhanced slice manager initialized successfully")
    
    def get_optimal_slice_allocation(self, current_state, return_all_steps=False):
        """Get optimal slice allocation based on current state
        
        This method uses the autoregressive LSTM predictor to predict optimal
        slice allocation for future steps.
        
        Args:
            current_state (numpy.ndarray): Current state vector
            return_all_steps (bool): Whether to return all predicted steps
            
        Returns:
            numpy.ndarray: Optimal slice allocation(s)
        """
        # Update history buffer
        self.update_history_buffer(current_state)
        
        # If we don't have enough history, use the base class implementation
        if len(self.history_buffer) < self.sequence_length:
            logger.info(f"Not enough history ({len(self.history_buffer)}/{self.sequence_length}), using default allocation")
            # Default allocation: equal distribution
            return np.array([1/3, 1/3, 1/3])
        
        # Convert history buffer to numpy array
        input_sequence = np.array(self.history_buffer)
        
        # Get prediction from autoregressive LSTM model
        prediction = self.predictor.predict(input_sequence, return_all_steps=return_all_steps)
        
        # Store prediction for analysis
        self.prediction_history.append(prediction if return_all_steps else np.array([prediction]))
        
        # Log prediction
        if return_all_steps:
            # For multi-step predictions, just log the shape
            logger.info(f"Multi-step prediction shape: {prediction.shape}")
        else:
            # Safe logging that handles numpy arrays correctly
            embb = prediction[0].item() if hasattr(prediction[0], 'item') else float(prediction[0])
            urllc = prediction[1].item() if hasattr(prediction[1], 'item') else float(prediction[1])
            mmtc = prediction[2].item() if hasattr(prediction[2], 'item') else float(prediction[2])
            logger.info(f"Predicted allocation: eMBB={embb:.2f}, URLLC={urllc:.2f}, mMTC={mmtc:.2f}")
        
        return prediction
    
    def train_model(self, training_data=None, epochs=150, batch_size=64, validation_split=0.2):
        """Train the autoregressive LSTM model
        
        Args:
            training_data (tuple): Training data (X, y)
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Validation split ratio
            
        Returns:
            keras.callbacks.History: Training history
        """
        if training_data is None:
            # Generate synthetic training data
            logger.info("Generating synthetic training data...")
            X_train, y_train, X_val, y_val = self.predictor._generate_training_data(15000)
        else:
            X, y = training_data
            # Split into training and validation sets
            split_idx = int((1 - validation_split) * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        logger.info("Training autoregressive LSTM model...")
        history = self.predictor.train(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            checkpoint_dir=self.checkpoint_dir
        )
        
        # Log training results
        val_loss = history.history['val_loss'][-1]
        val_mae = history.history['val_mae'][-1]
        logger.info(f"Training completed. Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")
        
        return history
    
    def save_model(self, path=None):
        """Save the autoregressive LSTM model
        
        Args:
            path (str): Path to save the model
        """
        if path is None:
            path = os.path.join(self.checkpoint_dir, f"autoregressive_lstm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
        
        self.predictor.save(path)
        logger.info(f"Model saved to {path}")
    
    def evaluate_model(self, test_data):
        """Evaluate the autoregressive LSTM model
        
        Args:
            test_data (tuple): Test data (X, y)
            
        Returns:
            tuple: (loss, mae)
        """
        X_test, y_test = test_data
        
        # Evaluate model
        loss, mae = self.predictor.evaluate(X_test, y_test)
        
        logger.info(f"Model evaluation: Loss={loss:.4f}, MAE={mae:.4f}")
        
        return loss, mae
    
    def visualize_predictions(self, input_sequence=None, actual_future=None):
        """Visualize predictions
        
        Args:
            input_sequence (numpy.ndarray): Input sequence
            actual_future (numpy.ndarray): Actual future values
        """
        if input_sequence is None and len(self.history_buffer) >= self.sequence_length:
            # Use history buffer as input sequence
            input_sequence = np.array(self.history_buffer)
        
        if input_sequence is not None:
            self.predictor.plot_predictions(input_sequence, actual_future)
        else:
            logger.warning("No input sequence available for visualization")


# Example usage
if __name__ == "__main__":
    # Create enhanced slice manager
    slice_manager = EnhancedSliceManager(input_dim=11, sequence_length=10, out_steps=5)
    
    # Generate sample data
    X_train, y_train, X_val, y_val = slice_manager.predictor._generate_training_data(1000)
    
    # Train model
    history = slice_manager.train_model(
        training_data=(X_train, y_train),
        epochs=50,
        batch_size=32,
        validation_split=0.2
    )
    
    # Visualize training history
    slice_manager.predictor.plot_training_history(history)
    
    # Make predictions
    sample_input = X_val[0]
    sample_output = y_val[0]
    
    # Get optimal slice allocation
    prediction = slice_manager.get_optimal_slice_allocation(sample_input[-1], return_all_steps=True)
    
    # Visualize predictions
    slice_manager.visualize_predictions(sample_input, sample_output)
    
    # Save model
    slice_manager.save_model("models/enhanced_slice_manager_model.h5") 