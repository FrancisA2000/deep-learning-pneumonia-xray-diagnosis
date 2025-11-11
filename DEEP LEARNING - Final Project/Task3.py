"""
Deep Learning Project - Pneumonia Detection from Chest X-rays
Task 3: Optimization Algorithms Comparison and Early Stopping
Author: [Your Name]
Date: June 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

# Reuse dataset loader and model builders from Task 1 (consistent with Task2New)
from Task1NEW import (
    load_datasets,
    create_cnn_without_transfer_learning,
    create_cnn_with_transfer_learning_frozen
)

# Suppress TensorFlow and Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)
import warnings
warnings.filterwarnings('ignore')  # Suppress Keras warnings

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# TensorFlow performance optimizations
tf.config.optimizer.set_jit(True)  # Enable XLA compilation

# Configure matplotlib for high-quality plots
plt.ioff()  # Turn off interactive mode
plt.style.use('default')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Create images directory for Task 3 outputs
IMAGES_DIR = os.path.join("images", "Task3")
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)
print(f"🖼️ Task 3 images will be saved under: {os.path.abspath(IMAGES_DIR)}")

def _fmt_lr(lr: float) -> str:
    """Format learning rate as plain decimal without scientific notation, trimming trailing zeros."""
    s = f"{lr:.8f}".rstrip('0').rstrip('.')
    return s

def _save_single_history_plot(history, arch_label: str, optimizer_name: str, epochs: int, lr: float):
    """Save a compact 2-panel plot (acc/loss) for a single training run."""
    try:
        # Save using the project-wide naming style under images/Task3
        lr_str = _fmt_lr(lr)
        filename = os.path.join(
            IMAGES_DIR,
            f"Task3_{arch_label}_{optimizer_name}_LR{lr_str}_Epochs{epochs}_Training_History.png"
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        epochs_range = range(1, len(history.history['accuracy']) + 1)

        # Accuracy
        ax1.plot(epochs_range, history.history['accuracy'], label='Train', color='#1f77b4')
        ax1.plot(epochs_range, history.history['val_accuracy'], label='Val', color='#ff7f0e', linestyle='--')
        ax1.set_title(f"Accuracy — {arch_label} | {optimizer_name} | ep={epochs} | lr={lr}")
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim([0, 1])
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        # Loss
        ax2.plot(epochs_range, history.history['loss'], label='Train', color='#1f77b4')
        ax2.plot(epochs_range, history.history['val_loss'], label='Val', color='#ff7f0e', linestyle='--')
        ax2.set_title('Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"🖼️ Run history saved: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"⚠️ Failed to save single history plot: {e}")

def train_with_optimizer(train_ds, val_ds, optimizer_name, epochs_list, learning_rates, model_factory, arch_label):
    """Train model with specific optimizer and different hyperparameters"""
    results = {}
    
    print(f"\n{'='*60}")
    print(f"[{arch_label}] TESTING {optimizer_name.upper()} OPTIMIZER")
    print(f"{'='*60}")
    
    for epochs in epochs_list:
        for lr in learning_rates:
            experiment_name = f"{arch_label}_{optimizer_name}_epochs{epochs}_lr{lr}"
            print(f"\n[{arch_label}] Training with {optimizer_name}, Epochs: {epochs}, LR: {lr}")
            
            # Create fresh model (from Task1NEW to keep consistency). Some factories
            # (e.g., TL variants) may return (model, base_model) — use the model.
            mf_res = model_factory()
            model = mf_res[0] if isinstance(mf_res, (tuple, list)) else mf_res
            
            # Set optimizer with current learning rate
            if optimizer_name == 'SGD':
                current_optimizer = SGD(learning_rate=lr)
            elif optimizer_name == 'SGD_Momentum':
                current_optimizer = SGD(learning_rate=lr, momentum=0.9)
            elif optimizer_name == 'Adam':
                current_optimizer = Adam(learning_rate=lr)
            elif optimizer_name == 'RMSprop':
                current_optimizer = RMSprop(learning_rate=lr)
            
            # Compile model
            model.compile(
                optimizer=current_optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                verbose=1
            )
            
            # Store results
            results[experiment_name] = {
                'history': history,
                'model': model,
                'final_train_acc': history.history['accuracy'][-1],
                'final_val_acc': history.history['val_accuracy'][-1],
                'final_train_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'max_val_acc': max(history.history['val_accuracy']),
                'min_val_loss': min(history.history['val_loss']),
                'epochs': epochs,
                'learning_rate': lr,
                'optimizer_name': optimizer_name,
                'arch_label': arch_label,
                'model_factory': model_factory,
            }
            
            print(f"✅ {experiment_name} - Max Val Acc: {max(history.history['val_accuracy']):.4f}")

            # Save a per-run history plot immediately so you see images during training
            _save_single_history_plot(history, arch_label, optimizer_name, epochs, lr)
            
            # Clear memory between runs
            tf.keras.backend.clear_session()
    
    return results

def train_with_early_stopping(train_ds, val_ds, best_params, model_factory, arch_label):
    """Train the best optimizer configuration with early stopping"""
    print(f"\n{'='*60}")
    print(f"[{arch_label}] TRAINING BEST OPTIMIZER WITH EARLY STOPPING")
    print(f"{'='*60}")
    
    # Clear session before creating a new model
    tf.keras.backend.clear_session()
    
    # Create model (support factories returning (model, base_model))
    mf_res = model_factory()
    model = mf_res[0] if isinstance(mf_res, (tuple, list)) else mf_res
    
    # Set up optimizer
    optimizer_name, lr = best_params['optimizer'], best_params['learning_rate']
    if optimizer_name == 'SGD':
        optimizer = SGD(learning_rate=lr)
    elif optimizer_name == 'SGD_Momentum':
        optimizer = SGD(learning_rate=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = RMSprop(learning_rate=lr)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=3,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )
    
    # Train with early stopping
    history_early_stopping = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,  # High number of epochs, early stopping will handle it
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history_early_stopping, model

def plot_optimizer_comparison(results_dict, arch_label):
    """Plot comparison of different optimizers and save as image"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    markers = ['o', 's', '^', 'd', 'x']  # Different markers for each optimizer
    
    # Training accuracy subplot
    for idx, (experiment_name, results) in enumerate(results_dict.items()):
        history = results['history']
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # Training and Validation Accuracy
        axes[0, 0].plot(history.history['accuracy'], 
                       label=f'{experiment_name} (Train)', 
                       color=color, linestyle='-', 
                       marker=marker, markersize=6,
                       alpha=0.8, linewidth=2.5)
        axes[0, 0].plot(history.history['val_accuracy'], 
                       label=f'{experiment_name} (Val)', 
                       color=color, linestyle='--', 
                       marker=marker, markersize=6,
                       alpha=0.8, linewidth=2.5)
    
    axes[0, 0].set_title(f'Training and Validation Accuracy Comparison\nDifferent Optimizers and Hyperparameters - {arch_label}', 
                         fontsize=16, fontweight='bold', pad=10)
    axes[0, 0].set_xlabel('Epoch', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    axes[0, 0].set_ylim([0, 1])
    
    # Loss subplot
    for idx, (experiment_name, results) in enumerate(results_dict.items()):
        history = results['history']
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # Training and Validation Loss
        axes[0, 1].plot(history.history['loss'], 
                       label=f'{experiment_name} (Train)', 
                       color=color, linestyle='-', 
                       marker=marker, markersize=6,
                       alpha=0.8, linewidth=2.5)
        axes[0, 1].plot(history.history['val_loss'], 
                       label=f'{experiment_name} (Val)', 
                       color=color, linestyle='--', 
                       marker=marker, markersize=6,
                       alpha=0.8, linewidth=2.5)
    
    axes[0, 1].set_title(f'Training and Validation Loss Comparison\nDifferent Optimizers and Hyperparameters - {arch_label}', 
                         fontsize=16, fontweight='bold', pad=10)
    axes[0, 1].set_xlabel('Epoch', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    # Final accuracy bar chart
    optimizer_names = []
    train_accs = []
    val_accs = []
    
    for experiment_name, results in results_dict.items():
        history = results['history']
        optimizer_names.append(experiment_name)
        train_accs.append(history.history['accuracy'][-1])
        val_accs.append(history.history['val_accuracy'][-1])
    
    x = np.arange(len(optimizer_names))
    width = 0.35
    
    train_bars = axes[1, 0].bar(x - width/2, train_accs, width, label='Training', color='#1f77b4')
    val_bars = axes[1, 0].bar(x + width/2, val_accs, width, label='Validation', color='#ff7f0e')
    
    # Add value labels to bars
    for bar in train_bars + val_bars:
        height = bar.get_height()
        axes[1, 0].annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')
    
    axes[1, 0].set_title(f'Final Accuracy Comparison by Optimizer - {arch_label}', 
                       fontsize=16, fontweight='bold', pad=10)
    axes[1, 0].set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(optimizer_names, rotation=45, ha='right', fontsize=12)
    axes[1, 0].legend(fontsize=12)
    axes[1, 0].grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[1, 0].set_ylim([0, 1.05])
    
    # Convergence speed bar chart (epochs to reach 90% of final val_acc)
    epochs_to_converge = []
    
    for experiment_name, results in results_dict.items():
        history = results['history']
        final_val_acc = history.history['val_accuracy'][-1]
        target_acc = 0.9 * final_val_acc
        
        # Find first epoch where val_acc >= 90% of final
        for i, acc in enumerate(history.history['val_accuracy']):
            if acc >= target_acc:
                epochs_to_converge.append(i + 1)  # +1 because epochs are 1-indexed
                break
        else:
            # If never reached, use total number of epochs
            epochs_to_converge.append(len(history.history['val_accuracy']))
    
    # Create horizontal bar chart for convergence speed
    bars = axes[1, 1].barh(optimizer_names, epochs_to_converge, color=colors)
    
    # Add value labels to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[1, 1].annotate(f'{width} epochs',
                           xy=(width, bar.get_y() + bar.get_height()/2),
                           xytext=(5, 0),  # 5 points horizontal offset
                           textcoords="offset points",
                           ha='left', va='center', 
                           fontsize=12, fontweight='bold')
    
    axes[1, 1].set_title(f'Convergence Speed Comparison - {arch_label}\nEpochs to Reach 90% of Final Accuracy', 
                        fontsize=16, fontweight='bold', pad=10)
    axes[1, 1].set_xlabel('Number of Epochs', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, axis='x', linestyle='--', alpha=0.6)
    axes[1, 1].invert_yaxis()  # To match order of other plots
    
    # Add text summary of best optimizer
    best_idx = np.argmax(val_accs)
    best_optimizer = optimizer_names[best_idx]
    best_acc = val_accs[best_idx]
    
    plt.figtext(0.5, 0.01, 
               f"Best Optimizer: {best_optimizer}\nValidation Accuracy: {best_acc:.4f}", 
               ha="center", fontsize=16, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.8, edgecolor="blue"))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    # Save the plot with minimal white space
    # Use naming style: Task3_{ARCH}_Optimizer_Comparison.png
    filename = os.path.join(IMAGES_DIR, f"Task3_{arch_label}_Optimizer_Comparison.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
        print(f"📊 Optimizer comparison saved: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"⚠️ Failed to save optimizer comparison: {e}")
    finally:
        plt.close()
    
    return filename

def plot_early_stopping_comparison(best_history, early_stopping_history, optimizer_name, arch_label):
    """Compare performance with and without early stopping and save as image"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # Colors and line styles for better visualization
    regular_color = '#1f77b4'  # Blue
    early_color = '#ff7f0e'    # Orange
    
    # Training Accuracy
    ax1.plot(best_history.history['accuracy'], 
            label='Without Early Stopping', 
            linewidth=3, color=regular_color,
            marker='o', markersize=6)
    
    ax1.plot(early_stopping_history.history['accuracy'], 
            label='With Early Stopping', 
            linewidth=3, color=early_color,
            marker='s', markersize=6)
    
    # Add vertical line at early stopping point
    early_stop_epoch = len(early_stopping_history.history['accuracy'])
    ax1.axvline(x=early_stop_epoch-1, color='red', linestyle='--', 
               label=f'Stopped at epoch {early_stop_epoch}')
    
    ax1.set_title(f'{optimizer_name} Optimizer - Training Accuracy ({arch_label})\nEarly Stopping vs Regular Training', 
                 fontsize=16, fontweight='bold', pad=10)
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Training Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim([0, 1])
    
    # Validation Accuracy
    ax2.plot(best_history.history['val_accuracy'], 
            label='Without Early Stopping', 
            linewidth=3, color=regular_color,
            marker='o', markersize=6)
    
    ax2.plot(early_stopping_history.history['val_accuracy'], 
            label='With Early Stopping', 
            linewidth=3, color=early_color,
            marker='s', markersize=6)
    
    # Add vertical line at early stopping point
    ax2.axvline(x=early_stop_epoch-1, color='red', linestyle='--')
    
    # Mark maximum validation accuracy for both approaches
    max_reg_val_acc = max(best_history.history['val_accuracy'])
    max_reg_val_acc_epoch = best_history.history['val_accuracy'].index(max_reg_val_acc)
    
    max_early_val_acc = max(early_stopping_history.history['val_accuracy'])
    max_early_val_acc_epoch = early_stopping_history.history['val_accuracy'].index(max_early_val_acc)
    
    ax2.plot(max_reg_val_acc_epoch, max_reg_val_acc, '*', 
            color=regular_color, markersize=16)
    ax2.annotate(f'{max_reg_val_acc:.4f}', 
                xy=(max_reg_val_acc_epoch, max_reg_val_acc),
                xytext=(5, 5), textcoords='offset points', 
                fontsize=12, fontweight='bold')
    
    ax2.plot(max_early_val_acc_epoch, max_early_val_acc, '*', 
            color=early_color, markersize=16)
    ax2.annotate(f'{max_early_val_acc:.4f}', 
                xy=(max_early_val_acc_epoch, max_early_val_acc),
                xytext=(5, -15), textcoords='offset points', 
                fontsize=12, fontweight='bold')
    
    ax2.set_title(f'{optimizer_name} Optimizer - Validation Accuracy ({arch_label})\nEarly Stopping vs Regular Training', 
                 fontsize=16, fontweight='bold', pad=10)
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim([0, 1])
    
    # Training Loss
    ax3.plot(best_history.history['loss'], 
            label='Without Early Stopping', 
            linewidth=3, color=regular_color,
            marker='o', markersize=6)
    
    ax3.plot(early_stopping_history.history['loss'], 
            label='With Early Stopping', 
            linewidth=3, color=early_color,
            marker='s', markersize=6)
    
    # Add vertical line at early stopping point
    ax3.axvline(x=early_stop_epoch-1, color='red', linestyle='--')
    
    ax3.set_title(f'{optimizer_name} Optimizer - Training Loss ({arch_label})\nEarly Stopping vs Regular Training', 
                 fontsize=16, fontweight='bold', pad=10)
    ax3.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    # Validation Loss
    ax4.plot(best_history.history['val_loss'], 
            label='Without Early Stopping', 
            linewidth=3, color=regular_color,
            marker='o', markersize=6)
    
    ax4.plot(early_stopping_history.history['val_loss'], 
            label='With Early Stopping', 
            linewidth=3, color=early_color,
            marker='s', markersize=6)
    
    # Add vertical line at early stopping point
    ax4.axvline(x=early_stop_epoch-1, color='red', linestyle='--')
    
    # Mark minimum validation loss for both approaches
    min_reg_val_loss = min(best_history.history['val_loss'])
    min_reg_val_loss_epoch = best_history.history['val_loss'].index(min_reg_val_loss)
    
    min_early_val_loss = min(early_stopping_history.history['val_loss'])
    min_early_val_loss_epoch = early_stopping_history.history['val_loss'].index(min_early_val_loss)
    
    ax4.plot(min_reg_val_loss_epoch, min_reg_val_loss, '*', 
            color=regular_color, markersize=16)
    ax4.annotate(f'{min_reg_val_loss:.4f}', 
                xy=(min_reg_val_loss_epoch, min_reg_val_loss),
                xytext=(5, 5), textcoords='offset points', 
                fontsize=12, fontweight='bold')
    
    ax4.plot(min_early_val_loss_epoch, min_early_val_loss, '*', 
            color=early_color, markersize=16)
    ax4.annotate(f'{min_early_val_loss:.4f}', 
                xy=(min_early_val_loss_epoch, min_early_val_loss),
                xytext=(5, 15), textcoords='offset points', 
                fontsize=12, fontweight='bold')
    
    ax4.set_title(f'{optimizer_name} Optimizer - Validation Loss ({arch_label})\nEarly Stopping vs Regular Training', 
                 fontsize=16, fontweight='bold', pad=10)
    ax4.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Validation Loss', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    # Add summary text at the bottom
    summary_text = (
        f"Early Stopping Summary - {optimizer_name} Optimizer\n"
        f"• Stopped at epoch {early_stop_epoch} of {len(best_history.history['accuracy'])}\n"
        f"• Best validation accuracy: {max_early_val_acc:.4f} vs {max_reg_val_acc:.4f} (regular)\n"
        f"• Training time savings: {100 * (1 - early_stop_epoch/len(best_history.history['accuracy'])):.1f}%"
    )
    
    plt.figtext(0.5, 0.01, summary_text, ha="center", fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', fc='lightblue', alpha=0.8, edgecolor='blue'))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])    # Save the plot with minimal white space
    # Use naming style: Task3_{ARCH}_{OPT}_Early_Stopping_Comparison.png
    filename = os.path.join(IMAGES_DIR, f"Task3_{arch_label}_{optimizer_name}_Early_Stopping_Comparison.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
        print(f"📊 Early stopping comparison saved: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"⚠️ Failed to save early stopping comparison: {e}")
    finally:
        plt.close()
    
    return filename

def print_results_summary(all_results):
    """Print comprehensive results summary"""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Create summary table
    summary_data = []
    for exp_name, results in all_results.items():
        summary_data.append({
            'Experiment': exp_name,
            'Epochs': results['epochs'],
            'Learning Rate': results['learning_rate'],
            'Final Train Acc': results['final_train_acc'],
            'Final Val Acc': results['final_val_acc'],
            'Max Val Acc': results['max_val_acc'],
            'Final Train Loss': results['final_train_loss'],
            'Final Val Loss': results['final_val_loss'],
            'Min Val Loss': results['min_val_loss']
        })
    
    # Sort by max validation accuracy
    summary_data.sort(key=lambda x: x['Max Val Acc'], reverse=True)
    
    print(f"{'Rank':<4} {'Experiment':<20} {'Epochs':<7} {'LR':<8} {'Max Val Acc':<12} {'Final Val Acc':<14} {'Min Val Loss':<12}")
    print("-" * 90)
    
    for i, data in enumerate(summary_data[:10]):  # Show top 10
        print(f"{i+1:<4} {data['Experiment']:<20} {data['Epochs']:<7} {data['Learning Rate']:<8.6f} "
              f"{data['Max Val Acc']:<12.4f} {data['Final Val Acc']:<14.4f} {data['Min Val Loss']:<12.4f}")
      # Find best performer
    best_experiment = summary_data[0]
    print(f"\n🏆 BEST PERFORMING CONFIGURATION:")
    print(f"   Experiment: {best_experiment['Experiment']}")
    print(f"   Max Validation Accuracy: {best_experiment['Max Val Acc']:.4f}")
    print(f"   Final Validation Accuracy: {best_experiment['Final Val Acc']:.4f}")
    print(f"   Epochs: {best_experiment['Epochs']}")
    print(f"   Learning Rate: {best_experiment['Learning Rate']:.6f}")
    
    return best_experiment

def main():
    """Main execution function for Task 3"""
    print("=== Deep Learning Project - Pneumonia Detection ===")
    print("Task 3: Optimization Algorithms Comparison and Early Stopping")
    
    # Load datasets
    train_ds, val_ds, test_ds = load_datasets()
    
    # Define hyperparameters to test - OPTIMIZED for faster training
    epochs_list = [15, 20]  # Start with higher epochs for better accuracy
    learning_rates = [0.001, 0.0001]  # Focus on most effective learning rates
    
    # Define architectures to evaluate
    architectures = [
        ("CNN", create_cnn_without_transfer_learning),
        ("TL_Frozen", create_cnn_with_transfer_learning_frozen),
    ]

    # Collect global results across architectures to pick a single best run overall
    global_results = {}

    for arch_label, model_factory in architectures:
        print("\n" + "="*60)
        print(f"ARCHITECTURE: {arch_label}")
        print("="*60)

        # Test different optimizers for this architecture
        all_results = {}
        
        sgd_results = train_with_optimizer(train_ds, val_ds, 'SGD', epochs_list, learning_rates, model_factory, arch_label)
        all_results.update(sgd_results)
        # Save an incremental optimizer comparison so you see progress
        plot_optimizer_comparison(all_results, arch_label)

        sgd_momentum_results = train_with_optimizer(train_ds, val_ds, 'SGD_Momentum', epochs_list, learning_rates, model_factory, arch_label)
        all_results.update(sgd_momentum_results)
        plot_optimizer_comparison(all_results, arch_label)

        adam_results = train_with_optimizer(train_ds, val_ds, 'Adam', epochs_list, learning_rates, model_factory, arch_label)
        all_results.update(adam_results)
        plot_optimizer_comparison(all_results, arch_label)

        rmsprop_results = train_with_optimizer(train_ds, val_ds, 'RMSprop', epochs_list, learning_rates, model_factory, arch_label)
        all_results.update(rmsprop_results)

        # Plot comparison for this architecture
        plot_optimizer_comparison(all_results, arch_label)

        # Print comprehensive results for this architecture
        _ = print_results_summary(all_results)

        # Accumulate into global results
        global_results.update(all_results)

    # Select the single best configuration across all architectures/optimizers/learning rates
    print(f"\n{'='*60}")
    print("SELECTING BEST OVERALL CONFIGURATION FOR EARLY STOPPING")
    print(f"{'='*60}")

    best_overall = print_results_summary(global_results)
    best_exp_name = best_overall['Experiment']
    best_meta = global_results[best_exp_name]
    best_arch_label = best_meta.get('arch_label', 'UNKNOWN')
    best_optimizer_name = best_meta.get('optimizer_name')
    best_lr = best_meta.get('learning_rate')
    best_history = best_meta['history']
    best_factory = best_meta.get('model_factory')

    print(f"\nBest Overall: {best_exp_name}")
    print(f"Architecture: {best_arch_label}")
    print(f"Optimizer: {best_optimizer_name}, LR: {best_lr}")

    best_params = { 'optimizer': best_optimizer_name, 'learning_rate': best_lr }

    # Train best overall with early stopping
    early_stopping_history, early_stopping_model = train_with_early_stopping(
        train_ds, val_ds, best_params, best_factory, best_arch_label
    )

    # Compare and save plot
    plot_early_stopping_comparison(best_history, early_stopping_history, best_optimizer_name, best_arch_label)

    # Save model once for the overall best
    model_path = f"best_model_task3_overall.h5"
    early_stopping_model.save(model_path)
    print(f"\nBest overall model saved as '{model_path}'")

    print(f"\n{'='*60}")
    print("TASK 3 COMPLETED — EARLY STOPPING RUN FOR BEST OVERALL CONFIGURATION")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
