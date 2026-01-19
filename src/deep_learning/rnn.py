"""
Recurrent Neural Networks (RNNs) and LSTM
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleRNN(nn.Module):
    """A simple Recurrent Neural Network"""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out


class SimpleLSTM(nn.Module):
    """A simple LSTM network"""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out


def sequence_prediction_example():
    """Demonstrate RNN for sequence prediction"""
    # Generate synthetic sequence data
    # Task: predict next value in a sequence
    np.random.seed(42)
    
    def generate_sequence(length=100):
        """Generate a simple sequence"""
        t = np.linspace(0, 4*np.pi, length)
        sequence = np.sin(t) + 0.1 * np.random.randn(length)
        return sequence
    
    sequence = generate_sequence(200)
    
    # Prepare data: use past 10 values to predict next value
    seq_length = 10
    X, y = [], []
    
    for i in range(len(sequence) - seq_length):
        X.append(sequence[i:i+seq_length])
        y.append(sequence[i+seq_length])
    
    X = np.array(X).reshape(-1, seq_length, 1)
    y = np.array(y).reshape(-1, 1)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Create model
    model = SimpleRNN(input_size=1, hidden_size=32, output_size=1, num_layers=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training
    num_epochs = 100
    loss_history = []
    
    print("Training RNN for sequence prediction...")
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy()
    
    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original sequence
    axes[0].plot(sequence, label='Original Sequence', alpha=0.7)
    axes[0].set_title('Original Sequence')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot predictions vs actual
    axes[1].plot(y, label='Actual', alpha=0.7)
    axes[1].plot(predictions, label='Predicted', alpha=0.7)
    axes[1].set_title('Predictions vs Actual')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model, loss_history


def lstm_vs_rnn_comparison():
    """Compare LSTM and RNN performance"""
    # Generate synthetic data
    np.random.seed(42)
    
    def generate_sequence(length=100):
        t = np.linspace(0, 4*np.pi, length)
        sequence = np.sin(t) + 0.1 * np.random.randn(length)
        return sequence
    
    sequence = generate_sequence(200)
    
    # Prepare data
    seq_length = 10
    X, y = [], []
    
    for i in range(len(sequence) - seq_length):
        X.append(sequence[i:i+seq_length])
        y.append(sequence[i+seq_length])
    
    X = np.array(X).reshape(-1, seq_length, 1)
    y = np.array(y).reshape(-1, 1)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Train RNN
    rnn_model = SimpleRNN(input_size=1, hidden_size=32, output_size=1)
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.01)
    rnn_criterion = nn.MSELoss()
    
    rnn_losses = []
    print("Training RNN...")
    for epoch in range(50):
        rnn_outputs = rnn_model(X_tensor)
        rnn_loss = rnn_criterion(rnn_outputs, y_tensor)
        rnn_optimizer.zero_grad()
        rnn_loss.backward()
        rnn_optimizer.step()
        rnn_losses.append(rnn_loss.item())
    
    # Train LSTM
    lstm_model = SimpleLSTM(input_size=1, hidden_size=32, output_size=1)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)
    lstm_criterion = nn.MSELoss()
    
    lstm_losses = []
    print("Training LSTM...")
    for epoch in range(50):
        lstm_outputs = lstm_model(X_tensor)
        lstm_loss = lstm_criterion(lstm_outputs, y_tensor)
        lstm_optimizer.zero_grad()
        lstm_loss.backward()
        lstm_optimizer.step()
        lstm_losses.append(lstm_loss.item())
    
    # Compare losses
    plt.figure(figsize=(10, 5))
    plt.plot(rnn_losses, label='RNN', alpha=0.7)
    plt.plot(lstm_losses, label='LSTM', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RNN vs LSTM Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Final RNN Loss: {rnn_losses[-1]:.6f}")
    print(f"Final LSTM Loss: {lstm_losses[-1]:.6f}")
    
    return rnn_model, lstm_model, rnn_losses, lstm_losses


if __name__ == "__main__":
    print("=== Sequence Prediction with RNN ===")
    sequence_prediction_example()
    
    print("\n=== RNN vs LSTM Comparison ===")
    lstm_vs_rnn_comparison()
