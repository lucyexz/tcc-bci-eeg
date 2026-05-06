from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class CNNConfig:
    conv_filters: tuple[int, ...] = (32, 64)
    dropout_rate: float = 0.4
    dense_units: int = 64
    learning_rate: float = 0.0003
    batch_size: int = 128
    epochs: int = 100
    patience: int = 10
    validation_split: float = 0.2
    random_state: int = 42
    n_frames: int = 1
    use_class_weights: bool = True


@dataclass
class CNNResult:
    model: keras.Model
    history: dict
    y_pred: np.ndarray
    y_pred_proba: np.ndarray
    y_test: np.ndarray
    test_accuracy: float
    test_loss: float
    config: CNNConfig


def create_temporal_windows(X: np.ndarray, n_frames: int = 3) -> np.ndarray:
    """Stack n_frames consecutive samples as channels for temporal context."""
    n_samples = len(X) - n_frames + 1
    return np.stack([X[i:i + n_samples] for i in range(n_frames)], axis=1)
    # shape: (N - n_frames + 1, n_frames, 25)


def reshape_for_cnn(X: np.ndarray, n_frames: int = 3) -> np.ndarray:
    # X shape: (N, n_frames, 25) → (N, 5, 5, n_frames)
    return X.reshape(-1, 5, 5, n_frames)


def build_cnn_model(config: CNNConfig) -> keras.Sequential:
    hidden_layers = []
    if config.dense_units > 0:
        hidden_layers = [layers.Dense(config.dense_units, activation="relu")]

    model = keras.Sequential([
        layers.Input(shape=(5, 5, config.n_frames)),
        layers.Conv2D(config.conv_filters[0], (4, 4), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(config.conv_filters[1], (4, 4), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(config.dropout_rate),
        *hidden_layers,
        layers.Dense(3, activation="softmax"),
    ])
    return model


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: CNNConfig | None = None,
) -> CNNResult:
    if config is None:
        config = CNNConfig()

    tf.random.set_seed(config.random_state)
    np.random.seed(config.random_state)

    X_train_w = create_temporal_windows(X_train, config.n_frames)
    y_train_w = y_train[config.n_frames - 1:]

    X_test_w = create_temporal_windows(X_test, config.n_frames)
    y_test_w = y_test[config.n_frames - 1:]

    X_train_cnn = reshape_for_cnn(X_train_w, config.n_frames)
    X_test_cnn = reshape_for_cnn(X_test_w, config.n_frames)

    class_weight = None
    if config.use_class_weights:
        weights = compute_class_weight(
            "balanced",
            classes=np.unique(y_train_w),
            y=y_train_w,
        )
        class_weight = dict(enumerate(weights))

    model = build_cnn_model(config)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        restore_best_weights=True,
    )
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    history = model.fit(
        X_train_cnn,
        y_train_w,
        validation_split=config.validation_split,
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weight,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_w, verbose=0)

    y_pred_proba = model.predict(X_test_cnn, verbose=0)
    y_pred = y_pred_proba.argmax(axis=1)

    return CNNResult(
        model=model,
        history=history.history,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        y_test=y_test_w,
        test_accuracy=test_accuracy,
        test_loss=test_loss,
        config=config,
    )
