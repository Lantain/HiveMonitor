import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt



def lstm_v1(SEGMENT_SIZE: int, sequence_length: int, n_features: int, n_outputs: int):
    model = tf.keras.Sequential([
        LSTM(SEGMENT_SIZE*10, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.4),
        LSTM(SEGMENT_SIZE*5, activation='relu'),
        Dropout(0.2),
        Dense(SEGMENT_SIZE * 2, activation='relu'),
        Dense(n_outputs, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

def lstm_v2(SEGMENT_SIZE: int, sequence_length: int, n_features: int, n_outputs: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(sequence_length, n_features)),
        
        # Add dropout to input
        tf.keras.layers.Dropout(0.2),
        
        # Simpler LSTM with L2 regularization
        tf.keras.layers.LSTM(SEGMENT_SIZE * 4, 
                            kernel_regularizer=tf.keras.regularizers.l2(0.01),
                            return_sequences=True),
        tf.keras.layers.LSTM(SEGMENT_SIZE, 
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        
        # Add dropout before dense layer
        tf.keras.layers.Dropout(0.2),
        
        # Output layer
        tf.keras.layers.Dense(n_outputs, activation='sigmoid')
    ])

    # Compile with reduced learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
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