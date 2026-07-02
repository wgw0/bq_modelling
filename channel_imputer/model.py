"""Model architecture and tf.data input pipeline."""

import logging
from typing import List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM, BatchNormalization, Concatenate, Dense, Dropout, Input, Masking,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

logger = logging.getLogger(__name__)


def build_model(feature_dim: int, seq_length: int, num_classes: int,
                learning_rate: float) -> Model:
    """
    Dual-input model: a dense branch over aggregated journey features and an
    LSTM branch over the session's event sequence.  Integer labels with
    sparse categorical cross-entropy avoid materialising one-hot targets.
    """
    input_agg = Input(shape=(feature_dim,), name='aggregated_features')
    x_agg = Dense(256, activation='relu')(input_agg)
    x_agg = BatchNormalization()(x_agg)
    x_agg = Dropout(0.3)(x_agg)
    x_agg = Dense(128, activation='relu')(x_agg)
    x_agg = Dropout(0.2)(x_agg)

    input_seq = Input(shape=(seq_length, feature_dim), name='sequence_features')
    x_seq = Masking(mask_value=0.0)(input_seq)
    x_seq = LSTM(128, return_sequences=False)(x_seq)
    x_seq = Dropout(0.2)(x_seq)

    merged = Concatenate()([x_agg, x_seq])
    x_merged = Dense(128, activation='relu')(merged)
    x_merged = BatchNormalization()(x_merged)
    x_merged = Dropout(0.3)(x_merged)
    x_merged = Dense(64, activation='relu')(x_merged)
    output = Dense(num_classes, activation='softmax')(x_merged)

    model = Model(inputs=[input_agg, input_seq], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary(print_fn=lambda line: logger.info(line))
    return model


def default_callbacks() -> list:
    return [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                          min_lr=1e-6),
    ]


def densify_batch(agg_rows: sp.csr_matrix,
                  seq_mats: Sequence[Optional[sp.csr_matrix]],
                  max_seq_len: int, feature_dim: int):
    """Densify one batch of sparse samples into model-ready float32 arrays."""
    agg = np.asarray(agg_rows.todense(), dtype=np.float32)
    seq = np.zeros((len(seq_mats), max_seq_len, feature_dim), dtype=np.float32)
    for row, mat in enumerate(seq_mats):
        if mat is None or mat.shape[0] == 0:
            continue
        k = min(mat.shape[0], max_seq_len)
        seq[row, :k, :] = np.asarray(mat[:k].todense(), dtype=np.float32)
    return {'aggregated_features': agg, 'sequence_features': seq}


def make_dataset(agg_matrix: sp.csr_matrix,
                 seq_list: List[Optional[sp.csr_matrix]],
                 y: np.ndarray,
                 max_seq_len: int, feature_dim: int, batch_size: int,
                 shuffle: bool, seed: int) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset that densifies whole batches at a time.

    Batches are sliced from CSR matrices with vectorised scipy ops, so peak
    memory stays proportional to batch_size * feature_dim while avoiding the
    per-sample Python overhead of an element-wise generator.  Shuffling is a
    fresh full permutation per epoch (the generator is re-invoked each epoch),
    which removes the need for a large tf.data shuffle buffer.
    """
    n_samples = int(y.shape[0])
    rng = np.random.default_rng(seed)

    def _generator():
        order = rng.permutation(n_samples) if shuffle else np.arange(n_samples)
        for start in range(0, n_samples, batch_size):
            batch_idx = order[start:start + batch_size]
            inputs = densify_batch(
                agg_matrix[batch_idx],
                [seq_list[i] for i in batch_idx],
                max_seq_len, feature_dim,
            )
            yield inputs, y[batch_idx].astype(np.int32)

    output_signature = (
        {
            'aggregated_features': tf.TensorSpec(
                shape=(None, feature_dim), dtype=tf.float32),
            'sequence_features': tf.TensorSpec(
                shape=(None, max_seq_len, feature_dim), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(
        _generator, output_signature=output_signature
    )
    return ds.prefetch(tf.data.AUTOTUNE)
