import os
import tensorflow as tf
import kagglehub
import pathlib
import numpy as np

# Configuration
DATASET_URL = "vipoooool/new-plant-diseases-dataset"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SEED = 123

def download_dataset():
    """Downloads dataset using kagglehub if not found locally."""
    print("Searching for dataset...")
    # Common paths where it might be
    possible_paths = [
        "new_plant_diseases_dataset",
        "New Plant Diseases Dataset(Augmented)",
        os.path.join(os.getcwd(), "new_plant_diseases_dataset"),
        os.path.expanduser("~/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Check if it has content
            if len(os.listdir(path)) > 0:
                print(f"Dataset found at {path}")
                return path
            
    print("Dataset not found locally. Attempting download via kagglehub...")
    try:
        path = kagglehub.dataset_download(DATASET_URL)
        print(f"Dataset downloaded to {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have Kaggle API credentials set up or manually download the dataset.")
        return None

def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    """Splits the dataset into Train, Validation, and Test."""
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=SEED)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

def load_and_split_data():
    """Main function to load and split data."""
    data_path = download_dataset()
    if not data_path:
        raise ValueError("Dataset not found or could not be downloaded.")

    data_dir = pathlib.Path(data_path)
    
    # The dataset typically has 'New Plant Diseases Dataset(Augmented)' -> 'train' / 'valid'
    # We will try to target the 'train' folder as it contains all classes and most images.
    target_dir = data_dir
    
    # Traverse to find the directory containing class folders
    # Strategy: look for 'train' folder explicitly
    train_search = list(data_dir.rglob('train'))
    if train_search:
        target_dir = train_search[0]
        print(f"Found 'train' directory at: {target_dir}")
    else:
        print(f"Could not find specific 'train' directory, using root: {target_dir}")

    print(f"Loading data from {target_dir}...")
    
    # Load entire dataset
    full_dataset = tf.keras.utils.image_dataset_from_directory(
        target_dir,
        shuffle=True,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Get class names
    class_names = full_dataset.class_names
    print(f"Found {len(class_names)} classes: {class_names}")

    # Split
    train_ds, val_ds, test_ds = get_dataset_partitions_tf(full_dataset)
    
    return train_ds, val_ds, test_ds, class_names

# Data Augmentation Layer
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal"),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.RandomZoom(0.2),
])

def preprocess_datasets(train_ds, val_ds, test_ds):
    """Applies augmentation and prefetching."""
    # Augment only training data
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                            num_parallel_calls=tf.data.AUTOTUNE)
    
    # Buffer and prefetch
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    try:
        train_ds, val_ds, test_ds, classes = load_and_split_data()
        train_ds, val_ds, test_ds = preprocess_datasets(train_ds, val_ds, test_ds)
        
        print("\nData Prep Success!")
        print(f"Training Batches: {len(train_ds)}")
        print(f"Validation Batches: {len(val_ds)}")
        print(f"Test Batches: {len(test_ds)}")
        
        # Save class names for later
        with open('class_names.txt', 'w') as f:
            for name in classes:
                f.write(f"{name}\n")
        print("Class names saved to class_names.txt")
        
    except Exception as e:
        print(f"\nFAILED: {e}")
