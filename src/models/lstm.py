from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class LSTMConfig:
    lstm_units: tuple[int, ...] = (64, 32)
    bidirectional: bool = True
    dropout_rate: float = 0.4
    recurrent_dropout: float = 0.1
    dense_units: int = 64
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    patience: int = 10
    validation_split: float = 0.2
    random_state: int = 42
    use_class_weights: bool = True


@dataclass
class LSTMResult:
    model: keras.Model
    history: dict
    y_pred: np.ndarray
    y_pred_proba: np.ndarray
    y_test: np.ndarray
    test_accuracy: float
    test_loss: float
    config: LSTMConfig


@dataclass
class GRUConfig:
    gru_units: tuple[int, ...] = (64, 32)
    bidirectional: bool = True
    dropout_rate: float = 0.4
    recurrent_dropout: float = 0.1
    dense_units: int = 64
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    patience: int = 10
    validation_split: float = 0.2
    random_state: int = 42
    use_class_weights: bool = True


@dataclass
class GRUResult:
    model: keras.Model
    history: dict
    y_pred: np.ndarray
    y_pred_proba: np.ndarray
    y_test: np.ndarray
    test_accuracy: float
    test_loss: float
    config: GRUConfig


def reshape_for_lstm(X: np.ndarray) -> np.ndarray:
    """
    (N, 25) → (N, 5, 5): 5 bands as timesteps, 5 channels as features.

    Column layout is channel-major: [AF3_Theta…AF3_Gamma, AF4_Theta…Pz_Gamma].
    reshape(-1, 5, 5) gives (N, 5_channels, 5_bands).
    Transpose to (N, 5_bands, 5_channels) so LSTM processes frequency bands sequentially.
    """
    return X.reshape(-1, 5, 5).transpose(0, 2, 1)


def _wrap_bidirectional(rnn_layer: layers.Layer, bidirectional: bool) -> layers.Layer:
    if bidirectional:
        return layers.Bidirectional(rnn_layer)
    return rnn_layer


def build_lstm_model(config: LSTMConfig) -> keras.Sequential:
    n_layers = len(config.lstm_units)
    rnn_layers = []
    for i, units in enumerate(config.lstm_units):
        return_sequences = i < n_layers - 1
        lstm = layers.LSTM(
            units,
            return_sequences=return_sequences,
            recurrent_dropout=config.recurrent_dropout,
        )
        rnn_layers.append(_wrap_bidirectional(lstm, config.bidirectional))
        rnn_layers.append(layers.Dropout(config.dropout_rate))

    hidden_layers = []
    if config.dense_units > 0:
        hidden_layers = [
            layers.Dense(config.dense_units, activation="relu"),
            layers.Dropout(config.dropout_rate),
        ]

    model = keras.Sequential([
        layers.Input(shape=(5, 5)),
        *rnn_layers,
        *hidden_layers,
        layers.Dense(3, activation="softmax"),
    ])
    return model


def build_gru_model(config: GRUConfig) -> keras.Sequential:
    n_layers = len(config.gru_units)
    rnn_layers = []
    for i, units in enumerate(config.gru_units):
        return_sequences = i < n_layers - 1
        gru = layers.GRU(
            units,
            return_sequences=return_sequences,
            recurrent_dropout=config.recurrent_dropout,
        )
        rnn_layers.append(_wrap_bidirectional(gru, config.bidirectional))
        rnn_layers.append(layers.Dropout(config.dropout_rate))

    hidden_layers = []
    if config.dense_units > 0:
        hidden_layers = [
            layers.Dense(config.dense_units, activation="relu"),
            layers.Dropout(config.dropout_rate),
        ]

    model = keras.Sequential([
        layers.Input(shape=(5, 5)),
        *rnn_layers,
        *hidden_layers,
        layers.Dense(3, activation="softmax"),
    ])
    return model


def _fit(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    validation_split: float,
    use_class_weights: bool,
    random_state: int,
) -> tuple:
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    class_weight = None
    if use_class_weights:
        weights = compute_class_weight(
            "balanced",
            classes=np.unique(y_train),
            y=y_train,
        )
        class_weight = dict(enumerate(weights))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weight,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = y_pred_proba.argmax(axis=1)

    return history.history, y_pred, y_pred_proba, test_loss, test_accuracy


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: LSTMConfig | None = None,
) -> LSTMResult:
    if config is None:
        config = LSTMConfig()

    X_train_lstm = reshape_for_lstm(X_train)
    X_test_lstm = reshape_for_lstm(X_test)

    model = build_lstm_model(config)
    history, y_pred, y_pred_proba, test_loss, test_accuracy = _fit(
        model,
        X_train_lstm, y_train,
        X_test_lstm, y_test,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        epochs=config.epochs,
        patience=config.patience,
        validation_split=config.validation_split,
        use_class_weights=config.use_class_weights,
        random_state=config.random_state,
    )

    return LSTMResult(
        model=model,
        history=history,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        y_test=y_test,
        test_accuracy=test_accuracy,
        test_loss=test_loss,
        config=config,
    )


def train_gru(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: GRUConfig | None = None,
) -> GRUResult:
    if config is None:
        config = GRUConfig()

    X_train_gru = reshape_for_lstm(X_train)
    X_test_gru = reshape_for_lstm(X_test)

    model = build_gru_model(config)
    history, y_pred, y_pred_proba, test_loss, test_accuracy = _fit(
        model,
        X_train_gru, y_train,
        X_test_gru, y_test,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        epochs=config.epochs,
        patience=config.patience,
        validation_split=config.validation_split,
        use_class_weights=config.use_class_weights,
        random_state=config.random_state,
    )

    return GRUResult(
        model=model,
        history=history,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        y_test=y_test,
        test_accuracy=test_accuracy,
        test_loss=test_loss,
        config=config,
    )
