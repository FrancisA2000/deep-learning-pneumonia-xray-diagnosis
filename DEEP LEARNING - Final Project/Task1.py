"""
Deep Learning Project - Pneumonia Detection from Chest X-rays
Task 1: CNN Architecture Design and Model Definition

PROJECT STRUCTURE AND REPORT SECTIONS:
=======================================================
Task 1 addresses the following report sections:

SECTION 1: ARCHITECTURE DESIGN (Question 1)
- Part A: Design CNN without Transfer Learning
- Part B: Design CNN with Transfer Learning (2 approaches)

IMPLEMENTATION SOLUTIONS:
- Question 1a: create_cnn_without_transfer_learning() - Custom CNN architecture
- Question 1b: create_cnn_with_transfer_learning_frozen() - Frozen base model
- Question 1b: create_cnn_with_transfer_learning_finetuned() - Fine-tuned model

VISUALIZATION AND DOCUMENTATION:
- Dataset sample visualization for report
- Model architecture tables and diagrams
- Technical specifications for each network

OUTPUT FOR REPORT:
- Architecture comparison tables
- Sample dataset images
- Model parameter counts and layer details
=======================================================

This module implements two CNN architectures for pneumonia detection:
1. Custom CNN without Transfer Learning (4 Conv2D blocks + Dense layers)
2. CNN with Transfer Learning using ResNet152V2 pre-trained on ImageNet

Both models output binary classification probabilities (0=Normal, 1=Pneumonia)
"""

import os
import shutil
import importlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet152V2, VGG16, EfficientNetB0
from tensorflow.keras.optimizers import Adam

# =============================================================================
# SECTION 1 - CONFIGURATION AND SETUP
# Report Section: Project Setup and Data Preparation
# =============================================================================

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration parameters for all experiments
DATA_PATH = os.path.join("chest_xray", "chest_xray")
print(f"Using dataset path: {os.path.abspath(DATA_PATH)}")

LABEL_NAMES = ['NORMAL', 'PNEUMONIA']  # Binary classification labels
IMG_SIZE = (160, 160)  # Input image dimensions for CNN models
BATCH_SIZE = 32  # Training batch size

# Create images directory for Task 1 outputs
IMAGES_DIR = os.path.join("images", "Task1")
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# Dataset configuration parameters
ds_kwargs = {
    'class_names': LABEL_NAMES,
    'seed': 42,
    'image_size': IMG_SIZE,
    'batch_size': BATCH_SIZE
}

# =============================================================================
# SECTION 1 - DATA PREPROCESSING AND LOADING
# Report Section: Dataset Preparation for CNN Training
# =============================================================================

def preprocess_binary(image, label):
    """
    Data preprocessing for binary classification
    Report Section: Data Preprocessing Pipeline
    
    Normalizes pixel values from [0,255] to [0,1] range for optimal CNN training
    """
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def load_datasets():
    """
    Dataset loading and preprocessing pipeline
    Report Section: Dataset Loading
    
    Creates training, validation, and test datasets directly from folders:
    - Train: DATA_PATH/train
    - Val:   DATA_PATH/val
    - Test:  DATA_PATH/test
    """
    print("Loading datasets for binary classification ...")
    
    try:
        # Load train/val/test directly from their folders
        train_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(DATA_PATH, 'train'),
            **ds_kwargs
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(DATA_PATH, 'val'),
            **ds_kwargs
        )
        
        test_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(DATA_PATH, 'test'),
            **ds_kwargs
        )
        
        # Apply preprocessing pipeline to all datasets
        train_ds = train_ds.map(preprocess_binary, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(preprocess_binary, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(preprocess_binary, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Performance optimization with caching and prefetching
        train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        print("Datasets loaded successfully!")
        return train_ds, val_ds, test_ds
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None, None

# =============================================================================
# QUESTION 1A - CNN WITHOUT TRANSFER LEARNING
# Report Section: Custom CNN Architecture Design
# =============================================================================

def create_cnn_without_transfer_learning():
    """
    SOLUTION FOR QUESTION 1A: Custom CNN Architecture
    Report Section: Network 1 - CNN without Transfer Learning
    
    Creates a custom CNN model from scratch for binary pneumonia classification.
    
    ARCHITECTURE DETAILS:
    - Input Layer: 160x160x3 RGB chest X-ray images
    - Convolutional Blocks: 4 identical blocks with:
      * Conv2D: 32 filters, 3x3 kernel, ReLU activation
      * MaxPooling2D: 2x2 pool size, stride=2
    - Feature Extraction: Flatten layer converts 3D to 1D
    - Classification Head:
      * Dense: 128 neurons, ReLU activation
      * Output: 1 neuron, Sigmoid activation (binary classification)
    
    TRAINING CONFIGURATION:
    - Optimizer: Adam (default learning rate)
    - Loss Function: Binary Crossentropy
    - Metrics: Accuracy
    
    REPORT USAGE: Include architecture diagram and parameter count
    """
    
    model = Sequential([
        # Explicit input with fixed batch size for clearer diagrams
        tf.keras.Input(shape=(*IMG_SIZE, 3)),
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Fourth Convolutional Block
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Feature Extraction
        Flatten(),
        
        # Classification Head
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification output
    ])
    
    # Compile model with binary classification settings
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# =============================================================================
# QUESTION 1B - CNN WITH TRANSFER LEARNING (FROZEN APPROACH)
# Report Section: Network 2A - Transfer Learning with Frozen Base
# =============================================================================

def create_cnn_with_transfer_learning_frozen():
    """
    SOLUTION FOR QUESTION 1B (APPROACH 1): Transfer Learning with Frozen Base
    Report Section: Network 2A - CNN with Transfer Learning (Frozen)
    
    Creates CNN using pre-trained ResNet152V2 with frozen weights for feature extraction.
    
    ARCHITECTURE DETAILS:
    Base Network:
    - Model: ResNet152V2 pre-trained on ImageNet
    - Input Shape: 160x160x3 RGB images
    - Include Top: False (remove original classification head)
    - Trainable: False (frozen weights - no backpropagation)
    
    Custom Classification Head:
    - Flatten: Convert ResNet features to 1D vector
    - Dense: 128 neurons, ReLU activation
    - Output: 1 neuron, Sigmoid activation (binary classification)
    
    TRAINING CONFIGURATION:
    - Optimizer: Adam with low learning rate (0.0001)
    - Loss Function: Binary Crossentropy
    - Metrics: Accuracy
    
    ADVANTAGES:
    - Fast training (only classification head is trained)
    - Good for small datasets
    - Leverages ImageNet-learned features
    
    REPORT USAGE: Compare with custom CNN and fine-tuned approach
    """
    
    # Load pre-trained ResNet152V2 without classification head
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), batch_size=BATCH_SIZE)
    base_model = ResNet152V2(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    
    # Freeze all layers in the base model
    base_model.trainable = False
    
    # Create complete model with custom classification head
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile with low learning rate for transfer learning
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

# =============================================================================
# QUESTION 1B - CNN WITH TRANSFER LEARNING (FINE-TUNING APPROACH)
# Report Section: Network 2B - Transfer Learning with Fine-tuning
# =============================================================================

def create_cnn_with_transfer_learning_finetuned():
    """
    SOLUTION FOR QUESTION 1B (APPROACH 2): Transfer Learning with Fine-tuning
    Report Section: Network 2B - CNN with Transfer Learning (Fine-tuned)
    
    Creates CNN using pre-trained ResNet152V2 with fine-tuning for domain adaptation.
    
    ARCHITECTURE DETAILS:
    Base Network:
    - Model: ResNet152V2 pre-trained on ImageNet
    - Input Shape: 160x160x3 RGB images
    - Include Top: False (remove original classification head)
    - Fine-tuning Strategy: Unfreeze last layers (from layer 540 onwards)
    
    Custom Classification Head:
    - Flatten: Convert ResNet features to 1D vector
    - Dense: 128 neurons, ReLU activation
    - Output: 1 neuron, Sigmoid activation (binary classification)
    
    FINE-TUNING STRATEGY:
    - Phase 1: Train with frozen base (prevents catastrophic forgetting)
    - Phase 2: Unfreeze top layers and retrain with learning rate
    - Learning Rate: 0.0001 
    
    TRAINING CONFIGURATION:
    - Optimizer: Adam with learning rate (0.0001)
    - Loss Function: Binary Crossentropy
    - Metrics: Accuracy
    
    ADVANTAGES:
    - Better adaptation to chest X-ray domain
    - Higher potential accuracy than frozen approach
    - Balances between overfitting and underfitting
    
    REPORT USAGE: Compare performance with frozen and custom CNN approaches
    """
    
    # Load pre-trained ResNet152V2
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), batch_size=BATCH_SIZE)
    base_model = ResNet152V2(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )

    # Pure fine-tuning: unfreeze the base, but keep early layers frozen
    base_model.trainable = True
    fine_tune_at = 540  # Unfreeze from this layer onwards
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Create complete model with custom classification head
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile once with low learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

def display_model_architectures():
    """Display detailed architecture information for all three CNN models"""
    print("\n" + "="*80)
    print("📋 TASK 1: MODEL ARCHITECTURES")
    print("="*80)
    
    # Model 1: CNN without Transfer Learning
    print("\n🔸 Network 1: CNN without Transfer Learning")
    print("-" * 50)
    
    model1 = create_cnn_without_transfer_learning()
    model1.summary()
    
    print("\\nArchitecture Details:")
    print("• Input Layer: 160x160x3 RGB images")
    print("• Conv2D Block 1: 32 filters, 3x3 kernel, ReLU + MaxPool2D (2x2)")
    print("• Conv2D Block 2: 32 filters, 3x3 kernel, ReLU + MaxPool2D (2x2)")
    print("• Conv2D Block 3: 32 filters, 3x3 kernel, ReLU + MaxPool2D (2x2)")
    print("• Conv2D Block 4: 32 filters, 3x3 kernel, ReLU + MaxPool2D (2x2)")
    print("• Flatten Layer: Convert 3D to 1D")
    print("• Dense Layer: 128 neurons, ReLU activation")
    print("• Output Layer: 1 neuron, Sigmoid activation (binary classification)")
    print(f"• Total Parameters: {model1.count_params():,}")
      # Model 2A: Transfer Learning (Frozen)
    print("\n🔸 Network 2A: CNN with Transfer Learning (Frozen)")
    print("-" * 50)
    
    model2a, base_model = create_cnn_with_transfer_learning_frozen()
    
    print("\\nBase Model Details:")
    print("• Base Network: ResNet152V2")
    print("• Pre-trained Dataset: ImageNet")
    print("• Input Shape: 160x160x3")
    print("• Include Top: False (removed classification head)")
    print("• Trainable: False (frozen)")
    print(f"• Base Model Parameters: {base_model.count_params():,}")
    
    print("\\nAdded Layers:")
    print("• Flatten Layer: Convert base model output to 1D")
    print("• Dense Layer: 128 neurons, ReLU activation")
    print("• Output Layer: 1 neuron, Sigmoid activation")
    print(f"• Total Parameters: {model2a.count_params():,}")
    print(f"• Trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model2a.trainable_weights]):,}")
      # Model 2B: Transfer Learning (Fine-tuned)
    print("\n🔸 Network 2B: CNN with Transfer Learning (Fine-tuned)")
    print("-" * 50)
    
    model2b, base_model_ft = create_cnn_with_transfer_learning_finetuned()
    
    print("\\nFine-tuning Details:")
    print("• Base Network: ResNet152V2")
    print("• Fine-tune from layer: 540")
    print("• Learning Rate: 0.0001")
    print(f"• Total Parameters: {model2b.count_params():,}")
    print(f"• Trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model2b.trainable_weights]):,}")
    
    return model1, model2a, model2b

def _ensure_graphviz_path():
    """If 'dot' is not on PATH, append common Graphviz bin locations."""
    if shutil.which('dot') is None:
        candidates = [
            r"C:\\Program Files\\Graphviz\\bin",
            r"C:\\Program Files (x86)\\Graphviz\\bin",
            os.path.expandvars(r"%USERPROFILE%\\AppData\\Local\\Programs\\Graphviz\\bin"),
        ]
        for p in candidates:
            if os.path.isdir(p) and p not in os.environ.get('PATH', ''):
                os.environ['PATH'] = os.environ.get('PATH', '') + os.pathsep + p
                # Break after adding the first found path
                break

def _graphviz_available():
    """Return True if system 'dot' exists and pydot is importable.

    Note: tf.keras.utils.plot_model relies on pydot and Graphviz's 'dot'.
    The python 'graphviz' package is not strictly required.
    """
    _ensure_graphviz_path()
    has_dot = shutil.which('dot') is not None
    try:
        importlib.import_module('pydot')
        has_pydot = True
    except Exception:
        has_pydot = False
    return has_dot and has_pydot

def _for_plot_with_fixed_batch(model, batch_size=BATCH_SIZE):
    """Create a plotting-only clone with a fixed batch dimension.

    This does not modify the original model. It's used solely to make
    diagrams show the batch size (e.g., 32) instead of None.
    """
    try:
        # Support single-input models
        if isinstance(model.input_shape, list):
            # Fallback for unusual multi-input cases
            return model
        plot_inputs = tf.keras.Input(shape=model.input_shape[1:], batch_size=batch_size)
        plot_outputs = model(plot_inputs)
        return tf.keras.Model(plot_inputs, plot_outputs, name=f"{model.name}_plot")
    except Exception as e:
        print(f"ℹ️ Using original model for plot (fixed-batch clone failed: {e})")
        return model

def save_model_diagrams():
    """Save model architecture diagrams and tables for documentation"""
    try:
        # Ensure images directory for Task 1 exists
        if not os.path.exists(IMAGES_DIR):
            os.makedirs(IMAGES_DIR)
        
        print("\n📊 Saving model architecture diagrams and tables...")
        
        model1, model2a, model2b = display_model_architectures()

        if _graphviz_available():
            # Save model architectures as flow diagrams
            try:
                import re
                # Build the pydot graph, then replace leading None in labels for display
                dot = tf.keras.utils.model_to_dot(
                    model1,
                    show_shapes=True,
                    show_layer_names=True,
                    rankdir="TB",
                    expand_nested=True,
                )
                for node in dot.get_nodes():
                    attrs = node.get_attributes() or {}
                    lbl = attrs.get('label')
                    if isinstance(lbl, str) and 'None' in lbl:
                        new_lbl = re.sub(r"\(None\b", f"({BATCH_SIZE}", lbl)
                        if new_lbl != lbl:
                            node.set('label', new_lbl)
                cnn_arch_path = os.path.join(IMAGES_DIR, "Task1_CNN_Architecture.png")
                dot.write_png(cnn_arch_path)
                print(f"✅ CNN Architecture diagram saved: {cnn_arch_path}")
            except Exception as e:
                print(f"⚠️ Could not save CNN diagram: {e}")

            try:
                frozen_arch_path = os.path.join(IMAGES_DIR, "Task1_Transfer_Learning_Frozen_Architecture.png")
                tf.keras.utils.plot_model(
                    model2a,
                    to_file=frozen_arch_path, 
                    show_shapes=True,
                    show_layer_names=True,
                    rankdir="TB",
                )
                print(f"✅ Transfer Learning (Frozen) diagram saved: {frozen_arch_path}")
            except Exception as e:
                print(f"⚠️ Could not save Frozen diagram: {e}")

            try:
                finetuned_arch_path = os.path.join(IMAGES_DIR, "Task1_Transfer_Learning_Finetuned_Architecture.png")
                tf.keras.utils.plot_model(
                    model2b,
                    to_file=finetuned_arch_path,
                    show_shapes=True, 
                    show_layer_names=True,
                    rankdir="TB",
                )
                print(f"✅ Transfer Learning (Fine-tuned) diagram saved: {finetuned_arch_path}")
            except Exception as e:
                print(f"⚠️ Could not save Fine-tuned diagram: {e}")
        else:
            print("ℹ️ Skipping Graphviz diagrams (Graphviz/pydot not installed or 'dot' not on PATH).")
        
        # Create clear model architecture tables for Word documents
        print("\\n📋 Creating clear architecture tables for documentation...")
        
        create_model_architecture_table(
            model1, 
            "CNN without Transfer Learning",
            "Task1_CNN_Architecture_Table.png"
        )
        
        create_model_architecture_table(
            model2a,
            "CNN with Transfer Learning (Frozen)",
            "Task1_Transfer_Learning_Frozen_Table.png"
        )
        
        create_model_architecture_table(
            model2b,
            "CNN with Transfer Learning (Fine-tuned)",
            "Task1_Transfer_Learning_Finetuned_Table.png"
        )
        
    except Exception as e:
        print(f"⚠️ Could not save diagrams: {e}")

def visualize_dataset_samples():
    """Visualize sample images from the dataset for documentation"""
    print("\n📊 DATASET SAMPLE VISUALIZATION")
    print("="*50)
    
    try:
        # Load a small batch from validation dataset for visualization
        train_ds, val_ds, _ = load_datasets()
        
        if val_ds is None:
            print("❌ Could not load dataset for visualization")
            return
        
        # Take one batch for visualization
        sample_batch = val_ds.take(1)
        
        # Get images and labels from the batch
        for images, labels in sample_batch:
            # Create a figure with subplots, larger size for better visibility
            fig, axes = plt.subplots(2, 4, figsize=(16, 12))
            plt.style.use('default')
            
            # Title
            fig.suptitle('Validation Sample Images - Chest X-Ray Pneumonia Detection', 
                         fontsize=20, fontweight='bold', y=0.99)
              # Display 8 sample images (2 rows, 4 columns)
            for i in range(8):
                if i < len(images):
                    ax = axes[i // 4, i % 4]
                    
                    # Convert image to display format
                    img = images[i].numpy()
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    
                    # Use grayscale for better X-ray visualization
                    ax.imshow(img, cmap='gray')
                    
                    # Add label with enhanced styling
                    label_text = LABEL_NAMES[int(labels[i])]
                    color = 'green' if label_text == 'NORMAL' else 'red'
                    ax.set_title(f'{label_text}', color=color, fontsize=16, fontweight='bold', pad=5)
                    ax.axis('off')
                    
                    # Add border around the image for better visibility
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_color('black')
                        spine.set_linewidth(2)
                else:
                    axes[i // 4, i % 4].axis('off')
              # Add descriptive text below the plots
            plt.figtext(0.5, 0.01,
                       f"Dataset: {', '.join(LABEL_NAMES)} • Image Size: {IMG_SIZE[0]}×{IMG_SIZE[1]} • Batch Size: {BATCH_SIZE}", 
                       ha="center", fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout(rect=[0, 0.07, 1, 0.92])
            
            # Save the plot with high resolution
            sample_plot_path = os.path.join(IMAGES_DIR, "Task1_Sample_Dataset_Images.png")
            plt.savefig(sample_plot_path, dpi=300, bbox_inches='tight', pad_inches=0.2, 
                       facecolor='white')
            plt.close()
            
            print(f"✅ Sample images saved: {sample_plot_path}")
            print(f"📊 Showing VALIDATION samples from classes: {', '.join(LABEL_NAMES)}")
            print(f"🔍 Image shape: {IMG_SIZE}")
            print(f"📈 Batch size: {BATCH_SIZE}")
            break
            
    except Exception as e:
        print(f"⚠️ Could not create sample visualization: {e}")
        print("This might be due to dataset loading issues or missing data")

def create_model_architecture_table(model, title, filename):
    """Create a professional model architecture table as an image for documentation"""
    try:
        # Get model summary as text
        import io
        import sys
        import re
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        model.summary()
        summary_text = buffer.getvalue()
        sys.stdout = old_stdout
        
        # Parse the summary to extract layer information
        lines = summary_text.split('\n')
        
        # Find the table content
        table_lines = []
        in_table = False
        for line in lines:
            if '┏' in line or '┡' in line:
                in_table = True
                continue
            elif '└' in line:
                break
            elif in_table and ('│' in line or '├' in line):
                if '├' not in line:
                    table_lines.append(line)
        
        # Extract layer data
        layers_data = []
        for line in table_lines:
            if '│' in line:
                parts = line.split('│')
                if len(parts) >= 4:
                    layer_name = parts[1].strip()
                    output_shape = parts[2].strip()
                    param_count = parts[3].strip()
                    
                    if layer_name and layer_name != 'Layer (type)':
                        # Replace the leading None batch dimension with the configured batch size for display only
                        # Examples: (None, 158, 158, 32) -> (32, 158, 158, 32); (None, 2048) -> (32, 2048)
                        display_shape = re.sub(r"\(None\b", f"({BATCH_SIZE}", output_shape)
                        layers_data.append([layer_name, display_shape, param_count])
        
        # Create figure with proper size
        num_rows = len(layers_data)
        fig_height = max(9, 2 + num_rows * 0.4)
        fig, ax = plt.subplots(figsize=(16, fig_height))
        ax.axis('tight')
        ax.axis('off')
        
        # Headers
        headers = ['Layer (Type)', 'Output Shape', 'Parameters']
        
        # Create table with enhanced styling
        table = ax.table(cellText=layers_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.45, 0.35, 0.2])
        
        # Table styling
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.0, 3.2)
        
        # Header styling
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#1F4E79')
            cell.set_text_props(weight='bold', color='white', size=16)
            cell.set_height(0.15)
            cell.set_edgecolor('white')
            
        # Row styling - alternate row colors for better readability
        for i in range(1, len(layers_data) + 1):
            for j in range(len(headers)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#F2F2F2')
                else:
                    cell.set_facecolor('#E6F0FF')
                
                # Highlight parameter count cells
                if j == 2 and i > 0:
                    cell.set_text_props(weight='bold')
                    
                # Make layer names bold
                if j == 0:
                    cell.set_text_props(weight='bold')
                      # Add title
        plt.suptitle(f"CNN Architecture Table", 
                    fontsize=20, fontweight='bold', y=0.99)
                    
        # Add total parameters as a text box at the bottom
        total_params = sum([int(row[2].replace(',', '')) for row in layers_data])
        trainable_params = sum([int(row[2].replace(',', '')) for row in layers_data])
        
        plt.figtext(0.5, 0.01, 
                   f"Total Parameters: {total_params:,} | Trainable: {trainable_params:,}", 
                   ha="center", fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.6", facecolor="#1F4E79", alpha=0.8, edgecolor='black', color='white'))
        
        # Add tight layout with space for the title and footer text
        plt.subplots_adjust(top=0.92, bottom=0.05)
          # Save the plot
        filepath = os.path.join(IMAGES_DIR, filename)
        
        # Compact title with better positioning
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        
        # Parameter summary
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        param_text = f"Total Parameters: {total_params:,} | "
        param_text += f"Trainable: {trainable_params:,} | Non-trainable: {non_trainable_params:,}"
        
        # Position summary box closer to the table
        fig.text(0.5, 0.05, param_text, ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F4FD', 
                         edgecolor='#1F4E79', linewidth=2))
        
        # Minimize white space with tighter layout
        plt.subplots_adjust(top=0.95, bottom=0.10)
        
        # Save with high quality
        filepath = os.path.join(IMAGES_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.2,
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Enhanced model architecture table saved: {filepath}")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not create architecture table: {e}")
        return False

def main():
    """Main function for Task 1: Architecture Design and Model Creation"""
    print("\n" + "="*80)
    print("🚀 TASK 1: CNN ARCHITECTURE DESIGN")
    print("Task 1: Deep Learning Pneumonia Detection Model Design")
    print("="*80)
    
    # Load datasets to verify data availability
    train_ds, val_ds, test_ds = load_datasets()
    
    if train_ds is None:
        print("❌ Cannot proceed without datasets. Please check data path.")
        return
    
    # Display and create all model architectures
    model1, model2a, model2b = display_model_architectures()
      # Save architecture diagrams
    save_model_diagrams()
    
    # Visualize dataset samples
    visualize_dataset_samples()
    
    print("\n" + "="*80)
    print("✅ TASK 1 COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("📋 Summary:")
    print("• ✅ Network 1: CNN without Transfer Learning created")
    print("• ✅ Network 2A: CNN with Transfer Learning (Frozen) created")
    print("• ✅ Network 2B: CNN with Transfer Learning (Fine-tuned) created")
    print("• ✅ Architecture diagrams saved")
    print("• ✅ Architecture tables saved (perfect for Word documents)")
    print("• ✅ Sample dataset images saved")
    print("\n📝 Next Step: Run Task 2 for training and evaluation")
    print("="*80)
    
    return model1, model2a, model2b, train_ds, val_ds, test_ds

if __name__ == "__main__":
    main()
