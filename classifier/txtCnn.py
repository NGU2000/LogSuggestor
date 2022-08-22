from tensorflow.keras.layers import  *
from tensorflow.keras.models import Model
from tensorflow.keras import *
from utils.loss import  *
from configuration import voc_size, vec_len, max_len

class TextCNN:

    def __init__(self):
        # 深度模型测试
        self.vocad_size = voc_size
        self.vec_len = vec_len
        self.max_len = max_len

    def build_model(self, embeding_matrix = []):
        inputs = Input(shape = (self.max_len), name='img')
        x = Embedding(input_dim = self.vocad_size, output_dim = self.vec_len,
                      weights = [embeding_matrix],input_length = self.max_len)(inputs)
        x = Reshape((self.max_len, self.vec_len, 1), input_shape=(self.max_len, self.vec_len))(x)
        x1 = Conv2D(filters=32, kernel_size=(5, self.vec_len), padding='valid', activation='relu')(x)
        x1 = Dropout(0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = GlobalMaxPool2D()(x1)
        x2 = Conv2D(filters=32, kernel_size=(4, self.vec_len), padding='valid', activation='relu')(x)
        x1 = Dropout(0.2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = GlobalMaxPool2D()(x2)
        x3 = Conv2D(filters=32, kernel_size=(3, self.vec_len), padding='valid', activation='relu')(x)
        x1 = Dropout(0.2)(x1)
        x3 = BatchNormalization()(x3)
        x3 = GlobalMaxPool2D()(x3)
        x = Concatenate(axis=1)([x1, x2, x3])
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs, name='txt_cnn')
        model.compile(optimizer='adam', loss = CB_loss,
                      metrics = ['accuracy'])
        # model.summary()
        return model
