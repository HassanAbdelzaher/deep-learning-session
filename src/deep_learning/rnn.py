"""
Recurrent Neural Networks (RNNs) and LSTM
Comprehensive module with visualizations and examples
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches


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


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_rnn_cell(save_path: Optional[str] = None):
    """Visualize a single RNN cell"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    input_circle = Circle((1, 4), 0.3, color='lightblue', ec='black', linewidth=2)
    ax.add_patch(input_circle)
    ax.text(1, 4, 'x_t', ha='center', va='center', fontsize=12, fontweight='bold')
    
    h_prev_circle = Circle((1, 2), 0.3, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(h_prev_circle)
    ax.text(1, 2, 'h_{t-1}', ha='center', va='center', fontsize=12, fontweight='bold')
    
    cell_rect = Rectangle((2.5, 2.5), 1.5, 1.5, color='lightyellow', ec='black', linewidth=2)
    ax.add_patch(cell_rect)
    ax.text(3.25, 3.25, 'RNN\nCell', ha='center', va='center', fontsize=11, fontweight='bold')
    
    h_curr_circle = Circle((5, 3.25), 0.3, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(h_curr_circle)
    ax.text(5, 3.25, 'h_t', ha='center', va='center', fontsize=12, fontweight='bold')
    
    output_circle = Circle((6.5, 3.25), 0.3, color='lightcoral', ec='black', linewidth=2)
    ax.add_patch(output_circle)
    ax.text(6.5, 3.25, 'y_t', ha='center', va='center', fontsize=12, fontweight='bold')
    
    arrow1 = FancyArrowPatch((1.3, 4), (2.5, 3.5), arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='blue')
    ax.add_patch(arrow1)
    ax.text(1.9, 3.8, 'W_x', fontsize=10, ha='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    arrow2 = FancyArrowPatch((1.3, 2), (2.5, 2.8), arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='green')
    ax.add_patch(arrow2)
    ax.text(1.9, 2.4, 'W_h', fontsize=10, ha='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    arrow3 = FancyArrowPatch((4, 3.25), (4.7, 3.25), arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='red')
    ax.add_patch(arrow3)
    
    arrow4 = FancyArrowPatch((5.3, 3.25), (6.2, 3.25), arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='purple')
    ax.add_patch(arrow4)
    
    loop_arrow = FancyArrowPatch((5, 3.55), (5, 4.5), (1, 4.5), (1, 2.3),
                                arrowstyle='->', mutation_scale=20, linewidth=2,
                                color='orange', linestyle='--', connectionstyle="arc3,rad=0.3")
    ax.add_patch(loop_arrow)
    ax.text(0.3, 3.5, 'h_t → h_{t+1}', fontsize=9, rotation=90, ha='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.text(3.25, 1.5, 'h_t = tanh(W_x·x_t + W_h·h_{t-1} + b)', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(1, 5)
    ax.set_title('RNN Cell Architecture', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_lstm_cell(save_path: Optional[str] = None):
    """Visualize LSTM cell architecture"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    input_circle = Circle((1, 5), 0.3, color='lightblue', ec='black', linewidth=2)
    ax.add_patch(input_circle)
    ax.text(1, 5, 'x_t', ha='center', va='center', fontsize=12, fontweight='bold')
    
    h_prev_circle = Circle((1, 3), 0.3, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(h_prev_circle)
    ax.text(1, 3, 'h_{t-1}', ha='center', va='center', fontsize=12, fontweight='bold')
    
    c_prev_circle = Circle((1, 1), 0.3, color='lightyellow', ec='black', linewidth=2)
    ax.add_patch(c_prev_circle)
    ax.text(1, 1, 'c_{t-1}', ha='center', va='center', fontsize=12, fontweight='bold')
    
    lstm_rect = Rectangle((2.5, 2), 2, 3, color='lightcoral', ec='black', linewidth=2)
    ax.add_patch(lstm_rect)
    ax.text(3.5, 3.5, 'LSTM\nCell', ha='center', va='center', fontsize=12, fontweight='bold')
    
    forget_gate = Circle((4, 1.5), 0.2, color='orange', ec='black', linewidth=1.5)
    ax.add_patch(forget_gate)
    ax.text(4, 1.5, 'f', ha='center', va='center', fontsize=9, fontweight='bold')
    
    input_gate = Circle((4, 2.5), 0.2, color='orange', ec='black', linewidth=1.5)
    ax.add_patch(input_gate)
    ax.text(4, 2.5, 'i', ha='center', va='center', fontsize=9, fontweight='bold')
    
    output_gate = Circle((4, 4.5), 0.2, color='orange', ec='black', linewidth=1.5)
    ax.add_patch(output_gate)
    ax.text(4, 4.5, 'o', ha='center', va='center', fontsize=9, fontweight='bold')
    
    h_curr_circle = Circle((6, 3.5), 0.3, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(h_curr_circle)
    ax.text(6, 3.5, 'h_t', ha='center', va='center', fontsize=12, fontweight='bold')
    
    c_curr_circle = Circle((6, 1.5), 0.3, color='lightyellow', ec='black', linewidth=2)
    ax.add_patch(c_curr_circle)
    ax.text(6, 1.5, 'c_t', ha='center', va='center', fontsize=12, fontweight='bold')
    
    output_circle = Circle((7.5, 3.5), 0.3, color='lightcoral', ec='black', linewidth=2)
    ax.add_patch(output_circle)
    ax.text(7.5, 3.5, 'y_t', ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.arrow(1.3, 5, 1, -0.5, head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=1.5)
    ax.arrow(1.3, 3, 1, 0, head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=1.5)
    ax.arrow(1.3, 1, 1, 0.5, head_width=0.1, head_length=0.1, fc='orange', ec='orange', linewidth=1.5)
    ax.arrow(4.5, 3.5, 1.2, 0, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
    ax.arrow(4.5, 1.5, 1.2, 0, head_width=0.1, head_length=0.1, fc='purple', ec='purple', linewidth=2)
    ax.arrow(6.3, 3.5, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    ax.text(3.5, 0.3, 'LSTM: Forget Gate (f), Input Gate (i), Output Gate (o)', ha='center',
           fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(0, 6)
    ax.set_title('LSTM Cell Architecture', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def visualize_sequence_unfolding(save_path: Optional[str] = None):
    """Visualize RNN unrolled through time"""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    time_steps = 4
    x_positions = [1 + i*3 for i in range(time_steps)]
    
    for t, x_pos in enumerate(x_positions):
        input_circle = Circle((x_pos, 4), 0.3, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(input_circle)
        ax.text(x_pos, 4, f'x_{t}', ha='center', va='center', fontsize=11, fontweight='bold')
        
        rnn_rect = Rectangle((x_pos-0.4, 2.5), 0.8, 0.8, color='lightgreen', ec='black', linewidth=2)
        ax.add_patch(rnn_rect)
        ax.text(x_pos, 2.9, 'RNN', ha='center', va='center', fontsize=9, fontweight='bold')
        
        output_circle = Circle((x_pos, 1), 0.3, color='lightcoral', ec='black', linewidth=2)
        ax.add_patch(output_circle)
        ax.text(x_pos, 1, f'y_{t}', ha='center', va='center', fontsize=11, fontweight='bold')
        
        if t > 0:
            arrow = FancyArrowPatch((x_positions[t-1]+0.4, 2.9), (x_pos-0.4, 2.9),
                                   arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
            ax.add_patch(arrow)
            ax.text((x_positions[t-1] + x_pos) / 2, 3.3, 'h', fontsize=9, ha='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax.arrow(x_pos, 3.7, 0, -0.3, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        ax.arrow(x_pos, 2.5, 0, -0.3, head_width=0.1, head_length=0.1, fc='purple', ec='purple')
    
    ax.text(6.5, 0.3, 'RNN Unrolled Through Time', ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(0, 5)
    ax.set_title('RNN Unfolded Through Time', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    print("=== Sequence Prediction with RNN ===")
    sequence_prediction_example()
    
    print("\n=== RNN vs LSTM Comparison ===")
    lstm_vs_rnn_comparison()
    
    print("\n=== Generating Visualizations ===")
    print("Run individual visualization functions to see graphs!")