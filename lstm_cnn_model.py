import tensorflow as tf

class LSTMMultiTask(tf.keras.Model):
    def __init__(self, num_genres, hidden_size=128, reg_weight=5.0):
        super(LSTMMultiTask, self).__init__()
        self.reg_weight = reg_weight


        ####
        # add a conv2D block here
            #conv layer
            #max pooling
            #batch norm
        #####
        self.epsilon = 1e-3
        self.convlayer_1 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu',
            padding='SAME',  use_bias=True, kernel_initializer='he_normal'
        )
        self.batch_norm1 = tf.keras.layers.BatchNormalization(epsilon=self.epsilon)
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

        ###after trying conv layer, you can try adding some more dense layers here
        # self.extra_dense_1 = tf.keras.layers.Dense(32, activation='relu')
        # self.extra_dense_2 = tf.keras.layers.Dense(16, activation='relu')
        self.genre_dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.genre_dense2 = tf.keras.layers.Dense(16, activation='relu')

        self.emotion_dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.emotion_dense2 = tf.keras.layers.Dense(16, activation='relu')
        
        self.genre_output = tf.keras.layers.Dense(
            units=num_genres,
            activation='softmax',
        )

        self.reg_output = tf.keras.layers.Dense(
            units=2,
            activation='sigmoid',
        )

    def call(self, inputs, is_testing=False):
        x = tf.reshape(inputs, [-1, 256, 32, 3])
        x = self.convlayer_1(x)
        x = tf.nn.max_pool(
            x,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        )
        x = self.batch_norm1(x)
        x = tf.reshape(x, [tf.shape(inputs)[0], 10, -1])
        x = self.masking(x)
        x = self.pre_dense(x)
        x = self.dropout1(x, training=not is_testing)
        x = self.layernorm1(x)

        x = self.lstm(x)

        x_shared = self.shared_dense(x)
        x_shared = self.dropout2(x_shared, training=not is_testing)
        x_shared = self.layernorm2(x_shared)

        # x_genre = self.extra_dense_1(x_shared)
        # x_genre = self.extra_dense_2(x_genre)
        x_genre = self.genre_dense1(x_shared)
        x_genre = self.genre_dense2(x_genre)
        genre_preds = self.genre_output(x_genre)
        x_emotion = tf.stop_gradient(x_shared)
        x_emotion = self.emotion_dense1(x_emotion)
        x_emotion = self.emotion_dense2(x_emotion)
        # x_emotion = self.extra_dense_1(x_emotion)
        # x_emotion = self.extra_dense_2(x_emotion)
        va_preds = self.reg_output(x_emotion)
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