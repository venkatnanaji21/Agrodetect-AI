import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from data_prep import load_and_split_data, preprocess_datasets

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINE = 10
LEARNING_RATE_FINE = 1e-5

def create_model(num_classes):
    """Creates the MobileNetV2 based model."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model, base_model

def plot_history(history, fine_tune_epoch=0):
    """Plots training history."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0, 1])
    if fine_tune_epoch > 0:
        plt.plot([fine_tune_epoch, fine_tune_epoch],
                 plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    if fine_tune_epoch > 0:
        plt.plot([fine_tune_epoch, fine_tune_epoch],
                 plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('training_history.png')
    print("Saved training_history.png")

def evaluate_model(model, test_ds, class_names):
    """Generates confusion matrix and classification report."""
    print("Evaluating model...")
    y_pred = []
    y_true = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
        
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:\n")
    print(report)
    with open('classification_report.txt', 'w') as f:
        f.write(report)
        
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")

def main():
    # 1. Load Data
    print("Loading Data...")
    train_ds, val_ds, test_ds, class_names = load_and_split_data()
    train_ds, val_ds, test_ds = preprocess_datasets(train_ds, val_ds, test_ds)
    
    NUM_CLASSES = len(class_names)
    
    # 2. Create Model
    print("Creating Model...")
    model, base_model = create_model(NUM_CLASSES)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    # Callbacks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    checkpoint_filepath = 'best_model.keras'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    
    # 3. Phase 1: Feature Extraction (Head only)
    print(f"Starting Phase 1 Training (Head) for {EPOCHS_HEAD} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=[tensorboard_callback, model_checkpoint_callback]
    )
    
    # 4. Phase 2: Fine-Tuning
    print("Unfreezing top layers for Fine-Tuning...")
    base_model.trainable = True
    
    # Freeze all layers except top 20
    fine_tune_at = len(base_model.layers) - 20
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE_FINE),
                  metrics=['accuracy'])
    
    model.summary()
    
    total_epochs = EPOCHS_HEAD + EPOCHS_FINE
    
    # Early Stopping for fine tuning
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    print(f"Starting Phase 2 Training (Fine-Tuning) for {EPOCHS_FINE} more epochs...")
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=history.epoch[-1],
        epochs=total_epochs,
        callbacks=[tensorboard_callback, model_checkpoint_callback, early_stopping]
    )
    
    # Merge histories
    # (Simple plot function will handle the split visualization)
    full_history = history
    for k in history_fine.history:
        if k in full_history.history:
            full_history.history[k] += history_fine.history[k]
            
    plot_history(full_history, fine_tune_epoch=EPOCHS_HEAD)
    
    # 5. Evaluate
    # Load best model for evaluation
    print("Loading best model for evaluation...")
    best_model = tf.keras.models.load_model(checkpoint_filepath)
    evaluate_model(best_model, test_ds, class_names)
    
    print("Training Complete!")

if __name__ == "__main__":
    main()
