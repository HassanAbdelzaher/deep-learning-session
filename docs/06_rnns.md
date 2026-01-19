# Recurrent Neural Networks (RNNs)

## Table of Contents
1. [Introduction](#introduction)
2. [RNN Architecture](#rnn-architecture)
3. [LSTM Networks](#lstm-networks)
4. [Sequence Processing](#sequence-processing)
5. [Applications](#applications)

## Introduction

RNNs are designed for sequential data where:
- Order matters (time series, text, speech)
- Previous inputs affect current output
- We need to remember past information

## RNN Architecture

### Basic RNN Cell

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches

def visualize_rnn_cell():
    """Visualize a single RNN cell"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Input
    input_circle = Circle((1, 4), 0.3, color='lightblue', ec='black', linewidth=2)
    ax.add_patch(input_circle)
    ax.text(1, 4, 'x_t', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Hidden state (previous)
    h_prev_circle = Circle((1, 2), 0.3, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(h_prev_circle)
    ax.text(1, 2, 'h_{t-1}', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # RNN cell
    cell_rect = Rectangle((2.5, 2.5), 1.5, 1.5, color='lightyellow', ec='black', linewidth=2)
    ax.add_patch(cell_rect)
    ax.text(3.25, 3.25, 'RNN\nCell', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Hidden state (current)
    h_curr_circle = Circle((5, 3.25), 0.3, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(h_curr_circle)
    ax.text(5, 3.25, 'h_t', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Output
    output_circle = Circle((5, 1.5), 0.3, color='lightcoral', ec='black', linewidth=2)
    ax.add_patch(output_circle)
    ax.text(5, 1.5, 'y_t', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrows
    arrow1 = FancyArrowPatch((1.3, 4), (2.5, 3.5), arrowstyle='->', mutation_scale=20, linewidth=2, color='blue')
    ax.add_patch(arrow1)
    ax.text(1.9, 3.8, 'W_x', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    arrow2 = FancyArrowPatch((1.3, 2), (2.5, 2.8), arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
    ax.add_patch(arrow2)
    ax.text(1.9, 2.4, 'W_h', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    arrow3 = FancyArrowPatch((4, 3.25), (4.7, 3.25), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    arrow4 = FancyArrowPatch((5, 2.95), (5, 1.8), arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax.add_patch(arrow4)
    
    # Recurrent connection (dashed)
    arrow5 = FancyArrowPatch((5.3, 3.25), (5.3, 6), arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='purple', linestyle='--')
    ax.add_patch(arrow5)
    arrow6 = FancyArrowPatch((5.3, 6), (1.3, 6), arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='purple', linestyle='--')
    ax.add_patch(arrow6)
    arrow7 = FancyArrowPatch((1.3, 6), (1.3, 2.3), arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='purple', linestyle='--')
    ax.add_patch(arrow7)
    ax.text(3.3, 6.2, 'Recurrent Connection', fontsize=10, ha='center', color='purple', fontweight='bold')
    
    # Formula
    ax.text(3.25, 1.5, 'h_t = tanh(W_x·x_t + W_h·h_{t-1} + b)', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_xlim(0, 6.5)
    ax.set_ylim(1, 6.5)
    ax.set_title('RNN Cell Architecture', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('docs/images/rnn_cell.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_rnn_cell()
```

### Unrolled RNN

```python
def visualize_unrolled_rnn():
    """Visualize RNN unrolled through time"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Time steps
    time_steps = 4
    x_positions = [1, 4, 7, 10]
    
    for t, x_pos in enumerate(x_positions):
        # Input
        input_circle = Circle((x_pos, 4), 0.25, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(input_circle)
        ax.text(x_pos, 4, f'x_{t}', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # RNN cell
        cell_rect = Rectangle((x_pos-0.4, 2.5), 0.8, 0.8, color='lightyellow', ec='black', linewidth=2)
        ax.add_patch(cell_rect)
        ax.text(x_pos, 2.9, 'RNN', ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Hidden state
        h_circle = Circle((x_pos, 1.5), 0.25, color='lightgreen', ec='black', linewidth=2)
        ax.add_patch(h_circle)
        ax.text(x_pos, 1.5, f'h_{t}', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Output
        output_circle = Circle((x_pos, 0.3), 0.25, color='lightcoral', ec='black', linewidth=2)
        ax.add_patch(output_circle)
        ax.text(x_pos, 0.3, f'y_{t}', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Arrows
        if t > 0:
            # Hidden state flow
            arrow = FancyArrowPatch((x_positions[t-1]+0.25, 1.5), (x_pos-0.25, 1.5),
                                   arrowstyle='->', mutation_scale=15, linewidth=2, color='purple')
            ax.add_patch(arrow)
        
        # Vertical arrows
        arrow1 = FancyArrowPatch((x_pos, 3.75), (x_pos, 3.3), arrowstyle='->', 
                                mutation_scale=15, linewidth=1.5, color='blue')
        ax.add_patch(arrow1)
        arrow2 = FancyArrowPatch((x_pos, 2.1), (x_pos, 1.75), arrowstyle='->', 
                                mutation_scale=15, linewidth=1.5, color='green')
        ax.add_patch(arrow2)
        arrow3 = FancyArrowPatch((x_pos, 1.25), (x_pos, 0.55), arrowstyle='->', 
                                mutation_scale=15, linewidth=1.5, color='red')
        ax.add_patch(arrow3)
    
    ax.text(5.5, 5, 'Time →', fontsize=12, fontweight='bold', ha='center')
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 5)
    ax.set_title('RNN Unrolled Through Time', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('docs/images/unrolled_rnn.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_unrolled_rnn()
```

## LSTM Networks

### LSTM Cell Structure

LSTMs solve the vanishing gradient problem with gates that control information flow.

```python
def visualize_lstm_cell():
    """Visualize LSTM cell with gates"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Input and previous hidden state
    input_circle = Circle((1, 7), 0.3, color='lightblue', ec='black', linewidth=2)
    ax.add_patch(input_circle)
    ax.text(1, 7, 'x_t', ha='center', va='center', fontsize=11, fontweight='bold')
    
    h_prev_circle = Circle((1, 5), 0.3, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(h_prev_circle)
    ax.text(1, 5, 'h_{t-1}', ha='center', va='center', fontsize=11, fontweight='bold')
    
    c_prev_circle = Circle((1, 3), 0.3, color='lightyellow', ec='black', linewidth=2)
    ax.add_patch(c_prev_circle)
    ax.text(1, 3, 'c_{t-1}', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Gates
    forget_gate = Rectangle((3, 6), 1.2, 0.6, color='lightcoral', ec='black', linewidth=2)
    ax.add_patch(forget_gate)
    ax.text(3.6, 6.3, 'Forget\nGate', ha='center', va='center', fontsize=9, fontweight='bold')
    
    input_gate = Rectangle((3, 4.5), 1.2, 0.6, color='lightblue', ec='black', linewidth=2)
    ax.add_patch(input_gate)
    ax.text(3.6, 4.8, 'Input\nGate', ha='center', va='center', fontsize=9, fontweight='bold')
    
    output_gate = Rectangle((3, 3), 1.2, 0.6, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(output_gate)
    ax.text(3.6, 3.3, 'Output\nGate', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Cell state
    cell_rect = Rectangle((5.5, 2.5), 1.5, 3, color='wheat', ec='black', linewidth=2)
    ax.add_patch(cell_rect)
    ax.text(6.25, 4, 'Cell\nState', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output
    h_curr_circle = Circle((8.5, 4), 0.3, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(h_curr_circle)
    ax.text(8.5, 4, 'h_t', ha='center', va='center', fontsize=11, fontweight='bold')
    
    c_curr_circle = Circle((8.5, 2), 0.3, color='lightyellow', ec='black', linewidth=2)
    ax.add_patch(c_curr_circle)
    ax.text(8.5, 2, 'c_t', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrows
    arrows = [
        ((1.3, 7), (3, 6.3), 'blue'),
        ((1.3, 5), (3, 6.3), 'green'),
        ((1.3, 7), (3, 4.8), 'blue'),
        ((1.3, 5), (3, 4.8), 'green'),
        ((1.3, 7), (3, 3.3), 'blue'),
        ((1.3, 5), (3, 3.3), 'green'),
        ((4.2, 6.3), (5.5, 4.5), 'red'),
        ((4.2, 4.8), (5.5, 3.5), 'blue'),
        ((1.3, 3), (5.5, 2.8), 'yellow'),
        ((7, 4), (8.2, 4), 'green'),
        ((7, 2.8), (8.2, 2), 'yellow')
    ]
    
    for (x1, y1), (x2, y2), color in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                               mutation_scale=15, linewidth=1.5, color=color, alpha=0.7)
        ax.add_patch(arrow)
    
    ax.text(4.75, 0.5, 'LSTM uses gates to control information flow:\n• Forget Gate: What to forget\n• Input Gate: What to remember\n• Output Gate: What to output',
           ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 8)
    ax.set_title('LSTM Cell Architecture', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('docs/images/lstm_cell.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_lstm_cell()
```

## Sequence Processing

### Sequence-to-Sequence Example

```python
def sequence_prediction_example():
    """Demonstrate RNN sequence prediction"""
    # Generate sequence data
    np.random.seed(42)
    t = np.linspace(0, 4*np.pi, 50)
    sequence = np.sin(t) + 0.1 * np.random.randn(50)
    
    # Simple prediction: predict next value
    predictions = sequence[1:] + np.random.normal(0, 0.05, 49)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Original sequence
    axes[0].plot(sequence, 'b-o', linewidth=2, markersize=4, label='Original Sequence', alpha=0.7)
    axes[0].set_xlabel('Time Step', fontsize=12)
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].set_title('Time Series Data', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Predictions vs actual
    axes[1].plot(sequence[1:], 'b-o', linewidth=2, markersize=4, label='Actual', alpha=0.7)
    axes[1].plot(predictions, 'r-s', linewidth=2, markersize=4, label='RNN Predictions', alpha=0.7)
    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('Sequence Prediction', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/images/sequence_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()

sequence_prediction_example()
```

## Applications

### Text Generation Visualization

```python
def text_generation_visualization():
    """Visualize how RNNs generate text"""
    # Simulated character probabilities
    chars = ['h', 'e', 'l', 'o', ' ']
    sequences = [
        ('h', [0.8, 0.1, 0.05, 0.03, 0.02]),
        ('he', [0.1, 0.05, 0.7, 0.1, 0.05]),
        ('hel', [0.05, 0.05, 0.8, 0.05, 0.05]),
        ('hell', [0.05, 0.05, 0.1, 0.7, 0.1]),
        ('hello', [0.1, 0.1, 0.1, 0.1, 0.6])
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    
    for idx, (context, probs) in enumerate(sequences):
        bars = axes[idx].bar(chars, probs, color=['blue', 'green', 'orange', 'red', 'purple'], alpha=0.7)
        axes[idx].set_title(f'Context: "{context}"', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Probability', fontsize=10)
        axes[idx].set_ylim(0, 1)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Highlight predicted character
        max_idx = np.argmax(probs)
        bars[max_idx].set_color('gold')
        bars[max_idx].set_edgecolor('black')
        bars[max_idx].set_linewidth(2)
    
    plt.suptitle('RNN Text Generation: Character Probabilities', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('docs/images/text_generation.png', dpi=150, bbox_inches='tight')
    plt.show()

text_generation_visualization()
```

## Practice Exercises

1. **RNN Forward Pass**: Given input `[1, 2, 3]`, weights `W_x=0.5`, `W_h=0.3`, bias `b=0.1`, and initial hidden state `h_0=0`, compute `h_1`, `h_2`, `h_3`.

2. **LSTM Gates**: Explain what each gate (forget, input, output) does in an LSTM cell.

3. **Sequence Length**: Why do RNNs struggle with very long sequences?

## Summary

- **RNNs** process sequential data by maintaining hidden state
- **LSTMs** solve vanishing gradients with gated mechanisms
- **Unrolling** shows how RNNs process sequences over time
- **Applications** include text generation, translation, time series prediction
- **Gates** in LSTMs control what to remember and forget

## Next Steps

- Learn about [Transformers](../docs/07_transformers.md)
- Study [Attention Mechanisms](../docs/08_attention.md)
