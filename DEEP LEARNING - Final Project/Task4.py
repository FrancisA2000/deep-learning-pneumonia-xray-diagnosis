"""
Deep Learning Project - Pneumonia Detection from Chest X-rays
Task 4: True Multi-class Classification (NORMAL, BACTERIAL, VIRAL)

IMPORTANT:
- The Kaggle dataset provides only NORMAL and PNEUMONIA folders.
- To comply with Task 4 (3 classes), we create deterministic pseudo-labels:
  Each PNEUMONIA image is assigned to BACTERIAL or VIRAL via a stable hash
  of its file path. This yields consistent labels across all splits.

What this script does:
1) Builds a NEW split from ALL Kaggle folders (train/val/test merged) to avoid leakage.
   Test set is enforced to have at least:
      NORMAL = 200
      BACTERIAL = 100
      VIRAL = 100
   (falls back gracefully if fewer exist).
2) Trains a REAL 3-class CNN (Softmax) over a grid of optimizers × LRs × epochs.
3) Selects the best model by max validation accuracy, saves plots, and produces a 3×3 confusion matrix.

This preserves your previous Task 3/4 functionality but fixes the core issue.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List, Tuple
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -----------------------------
# Global config
# -----------------------------
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
plt.ioff()
plt.style.use('default')

# Create images directory for Task 4 outputs
IMAGES_DIR = os.path.join("images", "Task4")
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# Path to the Kaggle dataset root (contains 'train', 'val', 'test' folders)
DATA_PATH = os.path.join("chest_xray", "chest_xray")
print(f"✅ Using dataset root: {os.path.abspath(DATA_PATH)}")

# True 3-class names required by the assignment
LABEL_NAMES = ['NORMAL', 'BACTERIAL', 'VIRAL']  # mapped to 0,1,2 respectively

IMG_SIZE = (160, 160)  # slightly larger for more detail (can raise to 224 if GPU memory allows)
BATCH_SIZE = 32

# -----------------------------
# Data utilities
# -----------------------------
def _glob_many(patterns: List[str]) -> List[str]:
    """Glob multiple patterns and merge results."""
    paths = []
    for p in patterns:
        paths.extend(tf.io.gfile.glob(p))
    return sorted(list(set(paths)))

def _label_from_pneumonia_filename(path: str):
    name = os.path.basename(path).lower()
    if 'bacteria' in name:
        return 1  # BACTERIAL
    if 'virus' in name or 'viral' in name:
        return 2  # VIRAL
    return None  # לא ברור - נטפל בהמשך

def list_all_files(root: str) -> Tuple[List[str], List[int]]:
    """
    Build file lists and 3-class integer labels (0 NORMAL, 1 BACTERIAL, 2 VIRAL)
    from ALL Kaggle folders (train/val/test merged). PNEUMONIA is split
    deterministically to BACTERIAL/VIRAL via file-path hashing.
    """
    # All NORMAL files across train/val/test
    normal = _glob_many([
        os.path.join(root, "*", "NORMAL", "*.jpeg"),
        os.path.join(root, "*", "NORMAL", "*.jpg"),
        os.path.join(root, "*", "NORMAL", "*.png"),
    ])

    # All PNEUMONIA files across train/val/test
    pneu = _glob_many([
        os.path.join(root, "*", "PNEUMONIA", "*.jpeg"),
        os.path.join(root, "*", "PNEUMONIA", "*.jpg"),
        os.path.join(root, "*", "PNEUMONIA", "*.png"),
    ])

       # לחלק ל-BACTERIAL/VIRAL לפי השם
    bact, viral, unknown = [], [], []
    for p in sorted(set(pneu)):
        y = _label_from_pneumonia_filename(p)
        if y == 1:
            bact.append(p)
        elif y == 2:
            viral.append(p)
        else:
            unknown.append(p)  # אם יש קובץ בלי מילת מפתח (נדיר)

    # בניית הפלט
    filepaths = []
    labels = []

    normal = sorted(set(normal))
    bact   = sorted(set(bact))
    viral  = sorted(set(viral))

    filepaths.extend(normal); labels.extend([0] * len(normal))
    filepaths.extend(bact);   labels.extend([1] * len(bact))
    filepaths.extend(viral);  labels.extend([2] * len(viral))

    print(f"📦 Total files found: NORMAL={len(normal)}, BACTERIAL={len(bact)}, VIRAL={len(viral)}")
    if unknown:
        print(f"⚠️  {len(unknown)} pneumonia files had unknown subtype (no 'bacteria'/'virus' in name).")

    return filepaths, labels

def stratified_split(
    paths: List[str], labels: List[int],
    test_counts=(200, 100, 100),
    val_ratio=0.2, seed=42
):
    """
    Create NEW splits:
      - Test: target minimum counts per class (NORMAL, BACTERIAL, VIRAL)
      - Remaining → Train/Val by val_ratio, stratified
    Falls back if class has fewer samples than target.
    """
    rng = random.Random(seed)
    by_class = {0: [], 1: [], 2: []}
    for p, y in zip(paths, labels):
        by_class[y].append(p)

    for c in by_class:
        rng.shuffle(by_class[c])

    # Build test set with requested minimums (subject to availability)
    want_normal, want_bact, want_viral = test_counts
    test_norm = by_class[0][:min(want_normal, len(by_class[0]))]
    test_bact = by_class[1][:min(want_bact, len(by_class[1]))]
    test_viral = by_class[2][:min(want_viral, len(by_class[2]))]

    test_paths = test_norm + test_bact + test_viral
    test_labels = [0]*len(test_norm) + [1]*len(test_bact) + [2]*len(test_viral)

    # Remove test from pools
    by_class[0] = by_class[0][len(test_norm):]
    by_class[1] = by_class[1][len(test_bact):]
    by_class[2] = by_class[2][len(test_viral):]

    # Train/Val from remaining, stratified by ratio
    train_paths, train_labels, val_paths, val_labels = [], [], [], []
    for c in (0, 1, 2):
        remain = by_class[c]
        n_val = int(round(len(remain) * val_ratio))
        val_c = remain[:n_val]
        train_c = remain[n_val:]
        val_paths.extend(val_c);   val_labels.extend([c]*len(val_c))
        train_paths.extend(train_c); train_labels.extend([c]*len(train_c))

    # Shuffle within splits for randomness (but deterministic)
    def _shuffle_in_place(ps, ys):
        tmp = list(zip(ps, ys))
        rng.shuffle(tmp)
        ps[:], ys[:] = zip(*tmp) if tmp else ([], [])

    _shuffle_in_place(train_paths, train_labels)
    _shuffle_in_place(val_paths, val_labels)
    _shuffle_in_place(test_paths, test_labels)

    print(f"🧪 NEW SPLITS:")
    print(f"  Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
    print(f"  Test per class: NORMAL={test_labels.count(0)}, BACTERIAL={test_labels.count(1)}, VIRAL={test_labels.count(2)}")
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

def decode_and_preprocess(path, label):
    """
    Read image, decode as RGB, resize, normalize to [0,1].
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # robust for .jpg/.jpeg
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def make_dataset(paths: List[str], labels: List[int], batch_size=BATCH_SIZE, training=False, cache=True):
    """
    Build a tf.data pipeline from file paths and integer labels.
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(10000, len(paths)), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if cache:
        ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def load_datasets_task4(
    data_path: str = DATA_PATH,
    test_counts=(200, 100, 100),
    val_ratio: float = 0.2,
    batch_size: int = BATCH_SIZE,
):
    """
    Dataset loading and preprocessing pipeline (Task 4 - Multi-class)
    Report Section: Dataset Loading

    Creates training, validation, and test datasets using a NEW split
    built from all folders under the Kaggle dataset root. The pipeline
    follows the clean style from Task 1: decode/resize → (cache) → prefetch.

    Returns: train_ds, val_ds, test_ds
    """
    print("Loading datasets for multi-class classification (NORMAL/BACTERIAL/VIRAL) ...")

    # Build a fresh split from all available data (deterministic and leak-free)
    all_paths, all_labels = list_all_files(data_path)
    (tr_paths, tr_labels), (va_paths, va_labels), (te_paths, te_labels) = stratified_split(
        all_paths, all_labels, test_counts=test_counts, val_ratio=val_ratio, seed=42
    )

    # Build tf.data pipelines (Task1-like: map → cache → prefetch)
    train_ds = make_dataset(tr_paths, tr_labels, batch_size=batch_size, training=True, cache=True)
    val_ds   = make_dataset(va_paths, va_labels, batch_size=batch_size, training=False, cache=True)
    # Use batch_size=1 for test to get per-sample predictions for confusion matrix
    test_ds  = make_dataset(te_paths, te_labels, batch_size=1, training=False, cache=True)

    print("Datasets loaded successfully!")
    return train_ds, val_ds, test_ds

# -----------------------------
# Model + training utilities
# -----------------------------
def create_cnn_model_multiclass():
    """Enhanced CNN with BatchNorm, Dropout & GlobalAveragePooling to push accuracy higher.
    (Still lightweight; for >93% strongly consider transfer learning with EfficientNet / MobileNet.)
    """
    model = Sequential([
        # Input block
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(*IMG_SIZE, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.15),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.20),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.30),

        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(3, activation='softmax')
    ])
    return model

def display_sample_images(ds, class_names, max_per_class=4, outfile=f"{IMAGES_DIR}/Task4_Sample_Dataset_Images.png"):
    """
    Save a small grid of samples per class.
    """
    plt.figure(figsize=(16, 12))
    samples = {i: [] for i in range(len(class_names))}
    for images, labels in ds.unbatch().take(500):  # scan some to find examples
        i = int(labels.numpy())
        if len(samples[i]) < max_per_class:
            samples[i].append(images.numpy())
        if all(len(samples[c]) >= max_per_class for c in samples):
            break

    rows = len(class_names)
    cols = max_per_class
    idx = 1
    for c, cname in enumerate(class_names):
        for j in range(cols):
            plt.subplot(rows, cols, idx)
            if j < len(samples[c]):
                plt.imshow(samples[c][j])
                plt.title(cname, fontsize=14, fontweight='bold')
                plt.axis('off')
            idx += 1

    plt.suptitle('Sample Chest X-Ray Images (Task 4: Multi-class)', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.close()
    print(f"📊 Sample images saved: {outfile}")

def plot_training_history(history, experiment_name, optimizer_name, lr, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_ran = len(acc)
    best_idx = int(np.argmax(val_acc))
    max_val_acc = float(val_acc[best_idx])

    # Create figure
    plt.style.use('seaborn-v0_8-darkgrid') if 'seaborn-v0_8-darkgrid' in plt.style.available else plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Accuracy subplot
    ax1 = axes[0]
    ax1.plot(range(1, epochs_ran+1), acc, label='Train Accuracy', linewidth=2.2, color='#1f77b4')
    ax1.plot(range(1, epochs_ran+1), val_acc, label='Val Accuracy', linewidth=2.2, color='#d62728')
    ax1.scatter(best_idx+1, max_val_acc, color='gold', edgecolor='black', zorder=5, s=120, label='Best Val Acc')
    ax1.axvline(best_idx+1, color='gold', linestyle='--', linewidth=1.3, alpha=0.85)
    ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax1.set_title(f"{experiment_name} | Optimizer={optimizer_name} | LR={lr} | Epochs Run={epochs_ran}", fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10, frameon=True)
    ax1.set_ylim(0, max(1.0, max(acc + val_acc) * 1.02))

    # Annotate best val accuracy
    ax1.annotate(f"Best Val Acc = {max_val_acc:.4f}\n(Epoch {best_idx+1})",
                 xy=(best_idx+1, max_val_acc), xycoords='data',
                 xytext=(best_idx+1 + max(1, epochs_ran*0.03), max_val_acc - 0.05),
                 textcoords='data', fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='gold', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gold', alpha=0.85))

    # Loss subplot
    ax2 = axes[1]
    ax2.plot(range(1, epochs_ran+1), loss, label='Train Loss', linewidth=2.2, color='#1f77b4')
    ax2.plot(range(1, epochs_ran+1), val_loss, label='Val Loss', linewidth=2.2, color='#d62728')
    best_loss_idx = int(np.argmin(val_loss))
    min_val_loss = float(val_loss[best_loss_idx])
    ax2.scatter(best_loss_idx+1, min_val_loss, color='limegreen', edgecolor='black', s=110, zorder=6, label='Min Val Loss')
    ax2.axvline(best_loss_idx+1, color='limegreen', linestyle='--', linewidth=1.2, alpha=0.8)
    # Annotate min val loss with arrow & box
    ax2.annotate(
        f"Min Val Loss = {min_val_loss:.4f}\n(Epoch {best_loss_idx+1})",
        xy=(best_loss_idx+1, min_val_loss), xycoords='data',
        xytext=(best_loss_idx+1 + max(1, epochs_ran*0.03), min_val_loss + (0.05 * (max(val_loss) - min(val_loss)))),
        textcoords='data', fontsize=10, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='limegreen', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='limegreen', alpha=0.85)
    )
    ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Loss (Sparse CCE)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, frameon=True)

    # Text summary box (bottom)
    summary_text = (
        f"Final Train Acc: {acc[-1]:.4f}\n"
        f"Final Val Acc: {val_acc[-1]:.4f}\n"
        f"Best Val Acc: {max_val_acc:.4f} (Epoch {best_idx+1})\n"
        f"Final Train Loss: {loss[-1]:.4f}\n"
        f"Final Val Loss: {val_loss[-1]:.4f}"
    )
    fig.text(0.995, 0.01, summary_text, ha='right', va='bottom', fontsize=10, family='monospace',
             bbox=dict(facecolor='#f0f0f0', edgecolor='#cccccc', boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.04, 0.98, 0.97])
    fig.suptitle('Training History', fontsize=18, fontweight='bold', y=0.995)
    filename = f"{IMAGES_DIR}/{experiment_name}_Training_History.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"📊 Plot saved: {filename}")
    return max_val_acc

def save_phase1_summary(sorted_exps, best_exp, outfile=f"{IMAGES_DIR}/Task4_Phase1_Experiment_Summary.png", top_n=15):
    """Save a clean table image summarizing Phase 1 experiments instead of printing rows.

    Columns: Rank, Optimizer, LR, Epochs (Planned), BestEpoch (Val), MaxValAcc, Note
    Best row highlighted in gold.
    """
    top_n = min(top_n, len(sorted_exps))
    if top_n == 0:
        return None
    # Build table data
    headers = ["Rank", "Optimizer", "LR", "Epochs", "BestEpoch", "MaxValAcc", "Note"]
    rows = []
    best_row_index = None
    for rank, (name, res) in enumerate(sorted_exps[:top_n], start=1):
        note = "BEST" if name == best_exp else ""
        if note == "BEST":
            best_row_index = rank - 1
        rows.append([
            rank,
            res['optimizer'],
            f"{res['lr']:.4g}",
            res['epochs'],
            res['best_epoch'],
            f"{res['max_val_acc']:.4f}",
            note
        ])

    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.4 * (top_n + 1)))
    ax.axis('off')
    title = f"Task 4 - Phase 1 Experiment Summary (Top {top_n})"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=14)
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    # Styling
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#4f4f4f')
        if row == 0:
            cell.set_facecolor('#2F5597')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            if best_row_index is not None and row - 1 == best_row_index:
                cell.set_facecolor('#FFD700')
                cell.set_text_props(fontweight='bold')
            elif (row % 2) == 0:
                cell.set_facecolor('#F2F2F2')
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"📊 Phase 1 summary table saved: {outfile}")
    return outfile

def generate_multiclass_confusion_matrix(model, test_ds):
    """
    Compute REAL 3×3 confusion matrix on the test set.
    """
    y_true, y_pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    acc = np.trace(cm) / np.sum(cm)

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title('Confusion Matrix (NORMAL vs BACTERIAL vs VIRAL)', fontsize=18, fontweight='bold', pad=15)
    plt.tight_layout()
    filename = f"{IMAGES_DIR}/Task4_Confusion_Matrix_Final.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.close()
    print(f"📊 Confusion Matrix saved: {filename}")
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))
    return cm, acc

# -----------------------------
# Main experiment
# -----------------------------
def main():
    print("=== Deep Learning Project - Pneumonia Detection ===")
    print("Task 4: TRUE Multi-class Classification (Normal, Bacterial, Viral)")
    print("NOTE: PNEUMONIA pseudo-split to BACTERIAL/VIRAL is deterministic by file path hash")

    # 1) Load datasets (Task1-like clean loader)
    train_ds, val_ds, test_ds = load_datasets_task4(
        data_path=DATA_PATH,
        test_counts=(200, 100, 100),  # required minimums
        val_ratio=0.2,
        batch_size=BATCH_SIZE,
    )

    # Show sample images
    display_sample_images(train_ds, LABEL_NAMES)

    # 3) Hyperparameters grid (Task 4 = repeat Task 3 process WITHOUT transfer learning)
    # Phase 1: test optimizers × learning rates × epochs WITHOUT early stopping.
    # Phase 2: take best optimizer config and apply EARLY STOPPING; compare improvement.
    LRS = [0.001, 0.0001]
    EPOCHS = [15, 20 ]  # moderate lengths for scouting phase (keeps runtime reasonable)

    optimizers_config = {
        'SGD':          lambda lr: SGD(learning_rate=lr),
        'SGD_Momentum': lambda lr: SGD(learning_rate=lr, momentum=0.9),
        'Adam':         lambda lr: Adam(learning_rate=lr),
        'RMSprop':      lambda lr: RMSprop(learning_rate=lr),
    }

    all_experiments = {}
    best_val_acc = -1.0
    best_exp = None

    print("\n" + "="*80)
    print("TESTING OPTIMIZERS × LRs × EPOCHS (Phase 1: NO EarlyStopping)")
    print("="*80)

    # ------------------ PHASE 1: brute-force search (no early stopping) ------------------
    for opt_name, opt_fn in optimizers_config.items():
        print("\n" + "="*60)
        print(f"TESTING {opt_name.upper()}")
        print("="*60)
        for lr in LRS:
            for epochs in EPOCHS:
                exp_name = f"Task4_{opt_name}_LR{lr}_Epochs{epochs}"
                print(f"\nTraining: {opt_name}, LR={lr}, Epochs={epochs}")
                model = create_cnn_model_multiclass()
                model.compile(optimizer=opt_fn(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)

                max_val_acc = max(history.history['val_accuracy'])
                best_epoch_local = int(np.argmax(history.history['val_accuracy'])) + 1
                all_experiments[exp_name] = {
                    'history': history,
                    'model': model,
                    'optimizer': opt_name,
                    'lr': lr,
                    'epochs': epochs,
                    'max_val_acc': max_val_acc,
                    'best_epoch': best_epoch_local
                }

                if max_val_acc > best_val_acc:
                    best_val_acc = max_val_acc
                    best_model = model
                    best_exp = exp_name
                    print(f"✅ New best: {best_exp} | Max Val Acc = {best_val_acc:.4f} | Best Epoch = {best_epoch_local}")

                plot_training_history(history, exp_name, opt_name, lr, epochs)

    # 4) Summary of top models
    # Build sorted experiments and save table image (no textual table printed)
    sorted_exps = sorted(all_experiments.items(), key=lambda x: x[1]['max_val_acc'], reverse=True)
    save_phase1_summary(sorted_exps, best_exp)

    # ------------------ PHASE 2: EarlyStopping on best optimizer config ------------------
    print("\n" + "="*80)
    print("PHASE 2: EARLY STOPPING RE-TRAINING FOR BEST OPTIMIZER CONFIG")
    print("="*80)
    best_meta = all_experiments[best_exp]
    best_optimizer_name = best_meta['optimizer']
    best_lr = best_meta['lr']
    # Choose longer max epochs for early stopping attempt
    early_max_epochs = 60
    opt_fn = optimizers_config[best_optimizer_name]
    es_model = create_cnn_model_multiclass()
    es_model.compile(optimizer=opt_fn(best_lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop_cb = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    print(f"Re-training with EarlyStopping: Optimizer={best_optimizer_name}, LR={best_lr}, MaxEpochs={early_max_epochs}")
    es_history = es_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=early_max_epochs,
        verbose=1,
        callbacks=[early_stop_cb]
    )
    es_val_accs = es_history.history['val_accuracy']
    es_best_acc = float(np.max(es_val_accs))
    es_best_epoch = int(np.argmax(es_val_accs)) + 1
    improved = es_best_acc > best_val_acc + 1e-6
    print(f"EarlyStopping best Val Acc = {es_best_acc:.4f} at epoch {es_best_epoch} | Improvement over Phase1: {improved}")

    # Plot ES history
    plot_training_history(es_history, f"{best_exp}_EarlyStopping", best_optimizer_name, best_lr, early_max_epochs)

    # Final model for evaluation = early stopped if improved else original best phase1 model
    final_model = es_model if improved else best_meta['model']

    # Improved Top-5 visualization (horizontal, ranked, clear highlight)
    top5 = sorted_exps[:5]
    if top5:
        plt.figure(figsize=(12, 7)); plt.style.use('ggplot')

        # Build labels with rank + key hyperparams
        labels = []
        accs = []
        best_key = best_exp
        for rank, (exp_key, res) in enumerate(top5, start=1):
            label = f"{rank}. {res['optimizer']} | LR={res['lr']} | Ep={res['epochs']}"
            if exp_key == best_key:
                label += "  ⭐"
            labels.append(label)
            accs.append(res['max_val_acc'])

        # Horizontal bar chart (reverse order so rank 1 at top)
        y_pos = np.arange(len(labels))[::-1]
        colors = ["#1f77b4"] * len(labels)
        best_index = [k for k, (ek, _) in enumerate(top5) if ek == best_key][0]
        colors[len(labels) - 1 - best_index] = "#FFD700"  # align with reversed order

        bars = plt.barh(y_pos, accs[::-1], color=colors)

        # Annotate accuracy inside / at end of each bar
        for bar, acc in zip(bars, accs[::-1]):
            w = bar.get_width()
            plt.text(w + 0.001, bar.get_y() + bar.get_height()/2, f"{acc:.4f}",
                     va='center', ha='left', fontsize=11, fontweight='bold')

        plt.yticks(y_pos, labels[::-1], fontsize=11)
        plt.xlabel('Validation Accuracy', fontsize=13, fontweight='bold')
        plt.xlim(0, max(accs) + 0.02)
        plt.title('Top 5 Models by Validation Accuracy', fontsize=16, fontweight='bold', pad=12)
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        plt.tight_layout()
        summary_file = f"{IMAGES_DIR}/Task4_Model_Comparison_Summary.png"
        plt.savefig(summary_file, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
        plt.close()
        print(f"📊 Model comparison summary saved: {summary_file}")
    else:
        print("⚠️ Not enough experiments to plot Top-5 summary.")

    # 5) REAL 3-class confusion matrix on the best model
    print("\n" + "="*60)
    print("GENERATING TRUE MULTI-CLASS CONFUSION MATRIX")
    print("="*60)
    generate_multiclass_confusion_matrix(final_model, test_ds)

    print("\n" + "="*60)
    print("TASK 4 COMPLETED SUCCESSFULLY (Phase 1 + Phase 2)!")
    print("="*60)

if __name__ == "__main__":
    main()
