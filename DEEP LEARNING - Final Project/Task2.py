"""
Deep Learning Project - Pneumonia Detection from Chest X-rays
Task 2: Training, Evaluation, and Precision-Recall Analysis
משימה 2: אימון הרשתות וניתוח ביצועים

Training both networks and performing comprehensive evaluation:
1. Train both CNN networks (with/without Transfer Learning)
2. Compare Frozen vs Fine-tuning for Transfer Learning
3. Precision-Recall analysis with thresholds 0.1 to 0.9 (steps of 0.05)
4. F-Score calculation and optimal threshold finding
"""

import os
# Reduce TensorFlow log verbosity (hide INFO). Must be set before importing tensorflow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from matplotlib.ticker import PercentFormatter

# Import models from Task 1
# Use the no-split dataset loader and same models defined in Task1NEW
from Task1NEW import (
    create_cnn_without_transfer_learning,
    create_cnn_with_transfer_learning_frozen,
    create_cnn_with_transfer_learning_finetuned,
    load_datasets
)

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Create images directory for Task 2 outputs
IMAGES_DIR = os.path.join("images", "Task2")
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# Configure matplotlib
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})

def train_model_with_history(model, train_ds, val_ds, model_name, epochs=10):
    """
    Train a model and return training history
    """
    print(f"\\n🏋️ Training {model_name}...")
    print(f"   ⚙️ Epochs: {epochs}")
    print(f"   📊 Optimizer: {model.optimizer.__class__.__name__}")
    print(f"   📈 Learning Rate: {model.optimizer.learning_rate.numpy()}")
    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )
    
    # Print final results
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    
    print(f"\\n📊 {model_name} Training Results:")
    print(f"   🎯 Final Training Accuracy: {final_train_acc:.4f}")
    print(f"   🎯 Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"   🏆 Best Validation Accuracy: {best_val_acc:.4f}")
    
    return history

def plot_training_history(history, model_name, save_name, opt_name=None, opt_lr=None):
    """
    Plot training history with enhanced formatting
    """
    # Create a larger figure with subplots stacked vertically
    plt.figure(figsize=(16, 14))
    
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    
    # Plot accuracy with a grid
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy', 
             linewidth=2.5, marker='o', markersize=6, color='#1f77b4')
    plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy', 
             linewidth=2.5, marker='s', markersize=6, color='#ff7f0e')
    
    # Highlight best accuracy
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    plt.plot(best_epoch, best_val_acc, '*', markersize=18, color='green', 
             label=f'Best Val: {best_val_acc:.2%} @ epoch {best_epoch}')
    # Vertical marker for best epoch
    plt.axvline(best_epoch, color='green', linestyle='--', alpha=0.3)
    # Show percent ticks
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    
    # Enhanced styling
    plt.legend(loc='lower right', fontsize=14, frameon=True, facecolor='white', edgecolor='gray')
    plt.ylabel('Accuracy (%)', fontsize=16, fontweight='bold')
    plt.ylim([max(0.0, min(min(history.history['accuracy']), min(history.history['val_accuracy']))*0.9), 1.05])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f"Training and Validation Accuracy\n{model_name}", 
              fontsize=16, fontweight='bold', pad=10)

    # Plot loss with a grid
    ax2 = plt.subplot(2, 1, 2)
    plt.plot(epochs_range, history.history['loss'], label='Training Loss', 
             linewidth=2.5, marker='o', markersize=6, color='#1f77b4')
    plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss', 
             linewidth=2.5, marker='s', markersize=6, color='#ff7f0e')
    
    # Highlight best loss
    best_val_loss = min(history.history['val_loss'])
    best_loss_epoch = history.history['val_loss'].index(best_val_loss) + 1
    plt.plot(best_loss_epoch, best_val_loss, '*', markersize=18, color='green', 
             label=f'Best Val Loss: {best_val_loss:.4f} @ epoch {best_loss_epoch}')
    # Vertical marker for best loss epoch
    plt.axvline(best_loss_epoch, color='green', linestyle='--', alpha=0.3)
    
    plt.legend(loc='upper right', fontsize=14, frameon=True, facecolor='white', edgecolor='gray')
    plt.ylabel('Binary Crossentropy', fontsize=16, fontweight='bold')
    plt.ylim([0, max(max(history.history['loss']), max(history.history['val_loss']))*1.1])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=16, fontweight='bold')
    plt.title(f"Training and Validation Loss\n{model_name}", 
              fontsize=16, fontweight='bold', pad=10)
    
    # Add experiment info text box with very prominent max val accuracy
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    # Determine optimizer name and learning rate (prefer explicit args; fallback to history.model)
    resolved_opt_name = None
    resolved_lr_val = None
    if opt_name is not None:
        resolved_opt_name = str(opt_name)
    if opt_lr is not None:
        try:
            resolved_lr_val = float(opt_lr)
        except Exception:
            # Keep as-is if not a plain float
            resolved_lr_val = opt_lr

    if resolved_opt_name is None or resolved_lr_val is None:
        try:
            if hasattr(history, 'model') and history.model is not None:
                opt = history.model.optimizer
                if resolved_opt_name is None:
                    resolved_opt_name = opt.__class__.__name__
                if resolved_lr_val is None:
                    lr_attr = getattr(opt, 'learning_rate', None) or getattr(opt, 'lr', None)
                    if lr_attr is not None:
                        try:
                            resolved_lr_val = float(tf.keras.backend.get_value(lr_attr))
                        except Exception:
                            try:
                                resolved_lr_val = float(lr_attr.numpy())
                            except Exception:
                                resolved_lr_val = lr_attr
        except Exception:
            pass

    opt_name_to_show = resolved_opt_name if resolved_opt_name is not None else "Unknown"
    if isinstance(resolved_lr_val, (float, int)):
        opt_lr_str = f"{resolved_lr_val:.6f}"
    else:
        opt_lr_str = str(resolved_lr_val) if resolved_lr_val is not None else "?"
    summary = (
        f"Best Val Acc: {best_val_acc:.2%} @ epoch {best_epoch} | "
        f"Final Train: {final_train_acc:.2%} | Final Val: {final_val_acc:.2%} | "
    f"Min Val Loss: {best_val_loss:.4f} @ epoch {best_loss_epoch} | "
    f"Optimizer: {opt_name_to_show} | LR: {opt_lr_str}"
    )
    plt.figtext(0.5, 0.01, summary,
                ha="center", fontsize=14, fontweight='bold',
                bbox={"facecolor":"#e8f4fd", "alpha":0.95, "pad":8, "boxstyle":"round,pad=1"})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])  # Adjust layout to make room for the text box
      # Save plot with minimal white space
    filename = f"{IMAGES_DIR}/Task2_{save_name}_Training_History.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.close()
    print(f"📊 Training history saved: {filename}")
    
    return filename

def evaluate_with_thresholds(model, test_ds, model_name):
    """
    Task 2.2: Precision-Recall analysis with different thresholds
    עבור שתי הרשתות ציירו את גרף הביצועים PRECISION-RECALL
    """
    print(f"\\n🔍 Evaluating {model_name} with different thresholds...")
    
    # Get predictions and true labels
    y_true = []
    y_pred_prob = []
    
    for images, labels in test_ds:
        y_true.extend(labels.numpy())
        predictions = model.predict(images, verbose=0)
        y_pred_prob.extend(predictions.flatten())
    
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    
    # Calculate precision and recall for different thresholds
    thresholds = np.arange(0.1, 0.95, 0.05)  # From 0.1 to 0.9 with steps of 0.05
    precisions = []
    recalls = []
    f1_scores = []
    
    print(f"   📊 Testing {len(thresholds)} thresholds: {thresholds}")
    
    for threshold in thresholds:
        # Apply threshold
        y_pred_binary = (y_pred_prob >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Find best threshold based on F1-score
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    best_precision = precisions[best_f1_idx]
    best_recall = recalls[best_f1_idx]
    
    print(f"\\n🏆 Best Results for {model_name}:")
    print(f"   🎯 Best Threshold: {best_threshold:.2f}")
    print(f"   📈 Best F1-Score: {best_f1:.4f}")
    print(f"   📊 Precision at Best F1: {best_precision:.4f}")
    print(f"   📊 Recall at Best F1: {best_recall:.4f}")
    
    return {
        'thresholds': thresholds,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'best_precision': best_precision,
        'best_recall': best_recall
    }

def plot_precision_recall_analysis(results_dict, model_names):
    """
    Plot Precision-Recall curves and F1-score analysis for all models
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # More vibrant color scheme
    
    # Plot 1: Precision-Recall Curves (axes in %)
    for i, (model_name, results) in enumerate(results_dict.items()):
        ax1.plot(results['recalls'], results['precisions'], 
                '-', color=colors[i], linewidth=3, label=model_name,
                marker='o', markersize=6)
        
        # Mark best F1 point
        best_idx = np.argmax(results['f1_scores'])
        ax1.plot(results['recalls'][best_idx], results['precisions'][best_idx], 
                '*', color=colors[i], markersize=18)
        ax1.annotate(f'F1: {results["best_f1"]:.3f}', 
                    xy=(results['recalls'][best_idx], results['precisions'][best_idx]),
                    xytext=(8, 0), textcoords='offset points', 
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
    
    ax1.set_xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax1.set_title('Precision-Recall Curves\nComparison of All Models', 
                 fontsize=16, fontweight='bold', pad=10)
    ax1.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([0, 1.05])
    ax1.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    
    # Plot 2: F1-Score vs Threshold
    for i, (model_name, results) in enumerate(results_dict.items()):
        ax2.plot(results['thresholds'], results['f1_scores'], 
                '-', color=colors[i], linewidth=3, label=model_name,
                marker='o', markersize=6)
        
        # Mark best F1 point
        best_idx = np.argmax(results['f1_scores'])
        ax2.plot(results['thresholds'][best_idx], results['f1_scores'][best_idx], 
                '*', color=colors[i], markersize=18)
        ax2.annotate(f'{results["best_f1"]:.3f}', 
                    xy=(results['thresholds'][best_idx], results['f1_scores'][best_idx]),
                    xytext=(5, 8), textcoords='offset points', 
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
    
    ax2.set_xlabel('Classification Threshold', fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
    ax2.set_title('F1-Score vs Threshold\nOptimal Threshold Selection', 
                 fontsize=16, fontweight='bold', pad=10)
    ax2.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim([0.1, 0.9])
    
    # Plot 3: Precision vs Threshold (precision in %)
    for i, (model_name, results) in enumerate(results_dict.items()):
        ax3.plot(results['thresholds'], results['precisions'], 
                '-', color=colors[i], linewidth=3, label=model_name,
                marker='o', markersize=6)
        
        # Mark best threshold point
        best_idx = np.argmax(results['f1_scores'])
        ax3.plot(results['thresholds'][best_idx], results['precisions'][best_idx], 
                '*', color=colors[i], markersize=18)
    
    ax3.set_xlabel('Classification Threshold', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax3.set_title('Precision vs Threshold', fontsize=16, fontweight='bold', pad=10)
    ax3.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_xlim([0.1, 0.9])
    ax3.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    
    # Plot 4: Recall vs Threshold (recall in %)
    for i, (model_name, results) in enumerate(results_dict.items()):
        ax4.plot(results['thresholds'], results['recalls'], 
                '-', color=colors[i], linewidth=3, label=model_name,
                marker='o', markersize=6)
        
        # Mark best threshold point
        best_idx = np.argmax(results['f1_scores'])
        ax4.plot(results['thresholds'][best_idx], results['recalls'][best_idx], 
                '*', color=colors[i], markersize=18)
    
    ax4.set_xlabel('Classification Threshold', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
    ax4.set_title('Recall vs Threshold', fontsize=16, fontweight='bold', pad=10)
    ax4.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.set_xlim([0.1, 0.9])
    ax4.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    
    # Add summary text of best models and thresholds
    summary_text = "Best F1-Scores by Model:\n"
    for i, (model_name, results) in enumerate(results_dict.items()):
        summary_text += f"• {model_name}: {results['best_f1']:.4f} (Threshold: {results['best_threshold']:.2f})\n"
    
    plt.figtext(0.5, 0.01, summary_text, ha="center", fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=1', fc='lightblue', alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    # Save high-resolution images with minimal white space BEFORE closing the figure
    filename_primary = f"{IMAGES_DIR}/Task2_Precision_Recall_Analysis.png"
    filename_all = f"{IMAGES_DIR}/Task2_Precision_Recall_Analysis_All_Models.png"
    plt.savefig(filename_primary, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.savefig(filename_all, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.close()

    print(f"📊 Precision-Recall analysis saved: {filename_primary}")
    print(f"📊 Comprehensive Precision-Recall analysis saved: {filename_all}")

    return filename_all

def create_comparison_summary(results_dict, histories_dict):
    """
    Create comprehensive comparison summary
    """
    print("\\n" + "="*80)
    print("📊 TASK 2: COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    
    # Training Performance Comparison
    print("\\n🏋️ TRAINING PERFORMANCE:")
    print("-" * 50)
    for model_name, history in histories_dict.items():
        best_val_acc = max(history.history['val_accuracy'])
        final_val_acc = history.history['val_accuracy'][-1]
        epochs_trained = len(history.history['val_accuracy'])
        print(f"{model_name:30} | Best Val Acc: {best_val_acc:.4f} | Final: {final_val_acc:.4f} | Epochs: {epochs_trained}")
    
    # Threshold Analysis Results
    print("\\n🎯 THRESHOLD ANALYSIS RESULTS:")
    print("-" * 50)
    for model_name, results in results_dict.items():
        print(f"{model_name:30} | Best F1: {results['best_f1']:.4f} | Threshold: {results['best_threshold']:.2f}")
        print(f"{'':30} | Precision: {results['best_precision']:.4f} | Recall: {results['best_recall']:.4f}")
    
    # Find best performing model
    best_model = max(results_dict.keys(), key=lambda k: results_dict[k]['best_f1'])
    print(f"\\n🏆 BEST PERFORMING MODEL: {best_model}")
    print(f"   📈 Best F1-Score: {results_dict[best_model]['best_f1']:.4f}")
    print(f"   🎯 Optimal Threshold: {results_dict[best_model]['best_threshold']:.2f}")
    
    # Transfer Learning Comparison
    print("\\n🔄 TRANSFER LEARNING COMPARISON:")
    print("-" * 50)
    if 'Transfer Learning (Frozen)' in results_dict and 'Transfer Learning (Fine-tuned)' in results_dict:
        frozen_f1 = results_dict['Transfer Learning (Frozen)']['best_f1']
        finetuned_f1 = results_dict['Transfer Learning (Fine-tuned)']['best_f1']
        
        if finetuned_f1 > frozen_f1:
            print("✅ Fine-tuning IMPROVED performance")
            print(f"   📈 Improvement: {finetuned_f1 - frozen_f1:.4f}")
            better_method = "Fine-tuning"
        else:
            print("❌ Fine-tuning did NOT improve performance")
            print(f"   📉 Difference: {frozen_f1 - finetuned_f1:.4f}")
            better_method = "Frozen"
        
        print(f"\\n🏆 Better Transfer Learning Method: {better_method}")
    
    return best_model

def main():
    """
    Main function for Task 2: Training and Evaluation
    """
    print("\\n" + "="*80)
    print("🚀 TASK 2: TRAINING AND EVALUATION")
    print("משימה 2: אימון הרשתות וניתוח ביצועים")
    print("="*80)
    
    # Load datasets
    print("\\n📂 Loading datasets...")
    train_ds, val_ds, test_ds = load_datasets()
    
    if train_ds is None:
        print("❌ Cannot proceed without datasets. Please check data path.")
        return
    
    # Create models
    print("\\n🏗️ Creating models...")
    model1 = create_cnn_without_transfer_learning()
    model2a, _ = create_cnn_with_transfer_learning_frozen()
    model2b, _ = create_cnn_with_transfer_learning_finetuned()
    
    models = {
        'CNN (No Transfer Learning)': model1,
        'Transfer Learning (Frozen)': model2a,
        'Transfer Learning (Fine-tuned)': model2b
    }
    
    # Train all models
    histories = {}
    for model_name, model in models.items():
        history = train_model_with_history(model, train_ds, val_ds, model_name, epochs=20)
        histories[model_name] = history
        
        # Plot training history
        save_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        # Resolve optimizer info from the model we trained (we didn't recompile in Task 2)
        try:
            opt = model.optimizer
            opt_n = opt.__class__.__name__
            lr_attr = getattr(opt, 'learning_rate', None) or getattr(opt, 'lr', None)
            lr_val = None
            if lr_attr is not None:
                try:
                    lr_val = float(tf.keras.backend.get_value(lr_attr))
                except Exception:
                    try:
                        lr_val = float(lr_attr.numpy())
                    except Exception:
                        lr_val = None
        except Exception:
            opt_n, lr_val = "Unknown", None

        plot_training_history(history, model_name, save_name, opt_name=opt_n, opt_lr=lr_val)
    
    # Evaluate models with different thresholds
    print("\\n" + "="*60)
    print("🔍 PRECISION-RECALL ANALYSIS")
    print("="*60)

    # Select the best Transfer Learning variant (Frozen vs Fine-tuned) based on best validation accuracy
    tl_variants = [
        ('Transfer Learning (Frozen)', histories['Transfer Learning (Frozen)']),
        ('Transfer Learning (Fine-tuned)', histories['Transfer Learning (Fine-tuned)'])
    ]
    tl_variants_scored = [
        (name, max(hist.history['val_accuracy'])) for name, hist in tl_variants
    ]
    best_tl_name, best_tl_score = max(tl_variants_scored, key=lambda x: x[1])
    print(f"🏆 Best Transfer Learning variant for PR analysis: {best_tl_name} (Max Val Acc: {best_tl_score:.4f})")

    # Only evaluate thresholds for: (1) CNN, (2) the best TL variant
    pr_models = {
        'CNN (No Transfer Learning)': models['CNN (No Transfer Learning)'],
        best_tl_name: models[best_tl_name]
    }

    results = {}
    for model_name, model in pr_models.items():
        results[model_name] = evaluate_with_thresholds(model, test_ds, model_name)

    # Create comprehensive plots for the two selected models
    plot_precision_recall_analysis(results, list(pr_models.keys()))

    # Generate summary (based on the evaluated models)
    best_model = create_comparison_summary(results, {k: histories[k] for k in pr_models.keys()})
    
    print("\\n" + "="*80)
    print("✅ TASK 2 COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("📋 Summary:")
    print("• ✅ All models trained and evaluated")
    print("• ✅ Precision-Recall analysis completed")
    print("• ✅ Optimal thresholds found")
    print("• ✅ Transfer learning methods compared")
    print(f"• 🏆 Best Model: {best_model}")
    print("\\n📁 All results saved in 'images' folder")
    print("="*80)
    
    return models, histories, results

if __name__ == "__main__":
    main()
