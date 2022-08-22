#lstm做分类器
import matplotlib.pyplot as plt
import tensorflow as tf
from gensim.models import Word2Vec
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential


def get_embeddings(data, minimum, dimensions, window):
    emb_model = Word2Vec(
        data.tolist(),
        min_count=minimum,
        vector_size=dimensions,
        workers=4,
        window=window,
        sg=1,
    )
    word_vectors = emb_model.wv
    return word_vectors


def gensim_to_keras_embedding(keyed_vectors, train_embeddings=False):
    # keyed_vectors = model.wv  # structure holding the result of training
    weights = keyed_vectors.vectors
    # index_to_key = (
    #     keyed_vectors.index_to_key
    # )  # which row in `weights` corresponds to which word?

    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=train_embeddings,
    )
    return layer

def LSTM(x_train,y_train,x_test,y_test):
    x_train_val = x_train
    y_train_val = y_train
    word2vec = get_embeddings(x_train,0, 200, 5)
    model = Sequential()
    model.add(gensim_to_keras_embedding(word2vec))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", Precision(), Recall()],
    )

    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=False)

    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_train_val, y_train_val))
    precisions_test = history.history["val_precision"]
    recalls_test = history.history["val_recall"]
    f_test = [(2 * p * r) / (p + r) for p in precisions_test for r in recalls_test]
    plt.plot(f_test)
    plt.plot(precisions_test)
    plt.plot(recalls_test)
    plt.ylabel("Metric")
    plt.xlabel("Epoch")
    plt.legend(["F-Measure", "Precision", "Recall"], loc="upper left")
    plt.show()
