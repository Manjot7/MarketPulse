import tensorflow as tf
from config.settings import EPOCHS, LEARNING_RATE


def build_model(input_shape):
    """
    Build and compile a Bidirectional LSTM model.
    Two BiLSTM layers followed by a Dense output.
    """
    tf.random.set_seed(1234)

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=50, activation="tanh", return_sequences=True)
        ),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=30, activation="tanh", return_sequences=False)
        ),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(units=16, activation="relu"),
        tf.keras.layers.Dense(units=1, activation="linear")
    ])

    model.compile(
        loss=tf.keras.losses.mean_squared_error,
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    )

    return model


def train(X_train, y_train, X_val, y_val):
    """
    Train the BiLSTM model and return the trained model with history.
    """
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    return model, history
