from collections.abc import Sequence

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM
from tensorflow.keras.optimizers import Adam


def build_lstm_classifier(
    window_size: int,
    n_features: int,
    n_classes: int = 2,
    lstm_units: int | Sequence[int] = 64,
    dense_units: int | None = None,
    dropout: float = 0.3,
    learning_rate: float = 0.001,
    bidirectional: bool = False,
) -> Sequential:
    units = _normalise_units(lstm_units)
    layers = [Input(shape=(window_size, n_features))]

    for index, unit_count in enumerate(units):
        return_sequences = index < len(units) - 1
        lstm_layer = LSTM(unit_count, return_sequences=return_sequences)
        if bidirectional:
            layers.append(Bidirectional(lstm_layer))
        else:
            layers.append(lstm_layer)
        if dropout > 0:
            layers.append(Dropout(dropout))

    if dense_units is not None and dense_units > 0:
        layers.append(Dense(dense_units, activation="relu"))
        if dropout > 0:
            layers.append(Dropout(dropout))

    layers.append(Dense(n_classes, activation="softmax"))
    model = Sequential(layers)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def _normalise_units(lstm_units: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(lstm_units, int):
        return (lstm_units,)
    units = tuple(int(value) for value in lstm_units)
    if not units:
        raise ValueError("At least one LSTM layer is required.")
    return units
