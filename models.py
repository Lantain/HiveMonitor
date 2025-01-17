import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import matplotlib.pyplot as plt



def lstm_v1(SEGMENT_SIZE: int, sequence_length: int, n_features: int, n_outputs: int):
    model = tf.keras.Sequential([
        Bidirectional(LSTM(SEGMENT_SIZE*10, return_sequences=True, input_shape=(sequence_length, n_features))),
        Dropout(0.4),
        LSTM(SEGMENT_SIZE*5, activation='relu'),
        Dropout(0.2),
        Dense(SEGMENT_SIZE * 2, activation='relu'),
        Dense(n_outputs, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

def lstm_v2(SEGMENT_SIZE: int, sequence_length: int, n_features: int, n_outputs: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(sequence_length, n_features)),
        
        # Batch normalization on input
        tf.keras.layers.BatchNormalization(),
        
        # First LSTM layer with reduced complexity
        tf.keras.layers.LSTM(SEGMENT_SIZE * 2, 
                           kernel_regularizer=tf.keras.regularizers.l2(0.01),
                           recurrent_regularizer=tf.keras.regularizers.l2(0.01),
                           return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Second LSTM layer
        tf.keras.layers.LSTM(SEGMENT_SIZE, 
                           kernel_regularizer=tf.keras.regularizers.l2(0.01),
                           recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Dense layer with reduced size
        tf.keras.layers.Dense(SEGMENT_SIZE // 2, 
                            activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer
        tf.keras.layers.Dense(n_outputs, activation='sigmoid')
    ])

    # Compile with reduced learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    return model

# Update early stopping to be more patient
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    min_delta=0.001  # Minimum change to qualify as an improvement
)

# Add learning rate reduction
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

def visualize_training(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()