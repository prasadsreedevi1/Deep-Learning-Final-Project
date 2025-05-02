import tensorflow as tf

class LSTMMultiTask(tf.keras.Model):
    def __init__(self, num_genres, hidden_size=128, reg_weight=5.0):
        super(LSTMMultiTask, self).__init__()
        self.reg_weight = reg_weight

        self.pre_dense = tf.keras.layers.Dense(256, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.layernorm1 = tf.keras.layers.LayerNormalization()

        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                hidden_size,
                return_sequences=False,
                recurrent_activation='sigmoid'
            ),
            merge_mode='concat'
        )

        self.shared_dense = tf.keras.layers.Dense(64, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.masking = tf.keras.layers.Masking(mask_value=0.0)

        self.genre_output = tf.keras.layers.Dense(
            units=num_genres,
            activation='softmax',
        )

        self.reg_output = tf.keras.layers.Dense(
            units=2,
            activation='linear',
        )



    def call(self, inputs, is_testing=False):
        x = self.masking(inputs)
        x = self.pre_dense(x)
        x = self.dropout1(x, training=not is_testing)
        x = self.layernorm1(x)

        x = self.lstm(x)

        x = self.shared_dense(x)
        x = self.dropout2(x, training=not is_testing)
        x = self.layernorm2(x)

        genre_preds = self.genre_output(x)
        va_preds = self.reg_output(x)
        return {"genre": genre_preds, "valence_arousal": va_preds}

    def compute_loss(self, preds, targets):
        genre_loss = tf.keras.losses.sparse_categorical_crossentropy(targets["genre"], preds["genre"])
        genre_loss = tf.reduce_mean(genre_loss)

        reg_loss = tf.keras.losses.mean_squared_error(targets["valence_arousal"], preds["valence_arousal"])
        reg_loss = tf.reduce_mean(reg_loss)

        return genre_loss + self.reg_weight * reg_loss

    def compute_metrics(self, preds, targets):
        genre_acc = tf.keras.metrics.sparse_categorical_accuracy(targets["genre"], preds["genre"])
        genre_acc = tf.reduce_mean(genre_acc)

        va_mae = tf.reduce_mean(tf.abs(targets["valence_arousal"] - preds["valence_arousal"]))

        return {"genre_accuracy": genre_acc, "valence_arousal_mae": va_mae}