import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Embedding, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class DataHandler(object):
    def __init__(self, word_max_length = 30, batch_size = 64, buffer_size = 20000):
        
        train_data, test_data = self._load_data()

        #Erstellt ein Dictionary vom Datenset
        #self.tokenizer_ru = tfds.features.text.SubwordTextEncoder.build_from_corpus((ru.numpy() for ru, en in train_data), target_vocab_size=2**13)
        self.soos = tfds.features.text.SubwordTextEncoder([])
        self.tokenizer_ru = self.soos.load_from_file("testen")
        #self.tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus((en.numpy() for ru, en in train_data), target_vocab_size=2**13)
        self.tokenizer_en = self.soos.load_from_file("testen2")


        #Aufbereitung der Daten
        self.train_data = self._prepare_training_data(train_data, word_max_length, batch_size, buffer_size)
        self.test_data = self._prepare_testing_data(test_data, word_max_length, batch_size)

        
    def _load_data(self):
        data, info = tfds.load('ted_hrlr_translate/ru_to_en', with_info=True, as_supervised=True)
        return data['train'], data['validation']
    
    def _prepare_training_data(self, data, word_max_length, batch_size, buffer_size):
        data = data.map(self._encode_tf_wrapper)
        data.filter(lambda x, y: tf.logical_and(tf.size(x) <= word_max_length, tf.size(y) <= word_max_length)) #Wörter die größer sind als 30 char müssen raus
        data = data.cache()
        data = data.shuffle(buffer_size).padded_batch(batch_size, padded_shapes=([-1], [-1]))  #Padding der Daten, sodass alle gleich lang sind
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        return data
        
    def _prepare_testing_data(self, data, word_max_length, batch_size): 
        data = data.map(self._encode_tf_wrapper)
        data = data.filter(lambda x, y: tf.logical_and(tf.size(x) <= word_max_length, tf.size(y) <= word_max_length)).padded_batch(batch_size, padded_shapes=([-1], [-1]))
        
    #Start und End token werden hinzugefügt
    def _encode(self, english, russian):
        russian = [self.tokenizer_ru.vocab_size] + self.tokenizer_ru.encode(russian.numpy()) + [self.tokenizer_ru.vocab_size+1]
        english = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(english.numpy()) + [self.tokenizer_en.vocab_size+1]
        return russian, english
    
    def _encode_tf_wrapper(self, pt, en):
        return tf.py_function(self._encode, [pt, en], [tf.int64, tf.int64])


#PositionalEncoding

class PositionalEncoding(object):
    def __init__(self, position, d):
        angle_rads = self._get_angles(np.arange(position)[:, np.newaxis], np.arange(d)[np.newaxis, :], d)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        self._encoding = np.concatenate([sines, cosines], axis=-1)
        self._encoding = self._encoding[np.newaxis, ...]
    
    def _get_angles(self, position, i, d):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d))
        return position * angle_rates
    
    def get_positional_encoding(self):
        return tf.cast(self._encoding, dtype=tf.float32)

data_container = DataHandler()

positional_encoding = PositionalEncoding(data_container.tokenizer_ru.vocab_size + 2, 128)

#positional_encoding = PositionalEncoding(50, 512) # eine reihe hat 512 einträge und es gibt 50 reihen
#positional_encoding_values = positional_encoding.get_positional_encoding()

class MaskHandler(object):
    def padding_mask(self, sequence):
        sequence = tf.cast(tf.math.equal(sequence, 0), tf.float32)
        return sequence[:, tf.newaxis, tf.newaxis, :]

    def look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask


#---------------
#Attention
#---------------
class ScaledDotProductAttentionLayer():
    def calculate_output_weights(self, q, k, v, mask):
        qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention = qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits = 0 
            scaled_attention_logits += (mask * -1e9)  

        weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(weights, v)

        return output, weights


class MultiHeadAttentionLayer(Layer):
    def __init__(self, num_neurons, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        
        self.num_heads = num_heads
        self.num_neurons = num_neurons
        self.depth = num_neurons // self.num_heads
        self.attention_layer = ScaledDotProductAttentionLayer()
        
        self.q_layer = Dense(num_neurons)
        self.k_layer = Dense(num_neurons)
        self.v_layer = Dense(num_neurons)

        self.linear_layer = Dense(num_neurons)

    def split(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # Run through linear layers
        q = self.q_layer(q)
        k = self.k_layer(k)
        v = self.v_layer(v)

        # Split the heads
        q = self.split(q, batch_size)
        k = self.split(k, batch_size)
        v = self.split(v, batch_size)

        # Run through attention
        attention_output, weights = self.attention_layer.calculate_output_weights(q, k, v, mask)
        
        # Prepare for the rest of processing
        output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.num_neurons))
        
        # Run through final linear layer
        output = self.linear_layer(concat_attention)

        return output, weights


#Positional Encoding + Embedding

class PreProcessingLayer(Layer):
    def __init__(self, num_neurons, vocabular_size):
        super(PreProcessingLayer, self).__init__()
        
        # Initialize
        self.num_neurons = num_neurons

        # Add embedings and positional encoding
        self.embedding = Embedding(vocabular_size, self.num_neurons)
        positional_encoding_handler = PositionalEncoding(vocabular_size, self.num_neurons)
        self.positional_encoding = positional_encoding.get_positional_encoding()

        # Add embedings and positional encoding
        self.dropout = Dropout(0.1)
    
    def call(self, sequence, training, mask):
        sequence_lenght = tf.shape(sequence)[1]
        sequence = self.embedding(sequence)
        
        sequence *= tf.math.sqrt(tf.cast(self.num_neurons, tf.float32))
        sequence += self.positional_encoding[:, :sequence_lenght, :]
        sequence = self.dropout(sequence, training=training)
        
        return sequence


#Layer

def build_multi_head_attention_layers(num_neurons, num_heads):
    multi_head_attention_layer = MultiHeadAttentionLayer(num_neurons, num_heads)   
    dropout = tf.keras.layers.Dropout(0.1)
    normalization = LayerNormalization(epsilon=1e-6)
    return multi_head_attention_layer, dropout, normalization

def build_feed_forward_layers(num_neurons, num_hidden_neurons):
    feed_forward_layer = tf.keras.Sequential()
    feed_forward_layer.add(Dense(num_hidden_neurons, activation='relu'))
    feed_forward_layer.add(Dense(num_neurons))
        
    dropout = Dropout(0.1)
    normalization = LayerNormalization(epsilon=1e-6)
    return feed_forward_layer, dropout, normalization


#Encoder

class EncoderLayer(Layer):
    def __init__(self, num_neurons, num_hidden_neurons, num_heads):
        super(EncoderLayer, self).__init__()

        # Build multi head attention layer and necessary additional layers
        self.multi_head_attention_layer, self.attention_dropout, self.attention_normalization = \
        build_multi_head_attention_layers(num_neurons, num_heads)   
            
        # Build feed-forward neural network and necessary additional layers
        self.feed_forward_layer, self.feed_forward_dropout, self.feed_forward_normalization = \
        build_feed_forward_layers(num_neurons, num_hidden_neurons)
       
    def call(self, sequence, training, mask):

        # Calculate attention output
        attnention_output, _ = self.multi_head_attention_layer(sequence, sequence, sequence, mask)
        attnention_output = self.attention_dropout(attnention_output, training=training)
        attnention_output = self.attention_normalization(sequence + attnention_output)
        
        # Calculate output of feed forward network
        output = self.feed_forward_layer(attnention_output)
        output = self.feed_forward_dropout(output, training=training)
        
        # Combine two outputs
        output = self.feed_forward_normalization(attnention_output + output)

        return output


#Decoderlayer

class DecoderLayer(Layer):
    def __init__(self, num_neurons, num_hidden_neurons, num_heads):
        super(DecoderLayer, self).__init__()

        # Build multi head attention layers and necessary additional layers
        self.multi_head_attention_layer1, self.attention_dropout1, self.attention_normalization1 =\
        build_multi_head_attention_layers(num_neurons, num_heads)   
        
        self.multi_head_attention_layer2, self.attention_dropout2, self.attention_normalization2 =\
        build_multi_head_attention_layers(num_neurons, num_heads)           

        # Build feed-forward neural network and necessary additional layers
        self.feed_forward_layer, self.feed_forward_dropout, self.feed_forward_normalization = \
        build_feed_forward_layers(num_neurons, num_hidden_neurons)

    def call(self, sequence, enconder_output, training, look_ahead_mask, padding_mask):

        attnention_output1, attnention_weights1 = self.multi_head_attention_layer1(sequence, sequence, sequence, look_ahead_mask)
        attnention_output1 = self.attention_dropout1(attnention_output1, training=training)
        attnention_output1 = self.attention_normalization1(sequence + attnention_output1)
        
        attnention_output2, attnention_weights2 = self.multi_head_attention_layer2(enconder_output, enconder_output, attnention_output1, padding_mask)
        attnention_output2 = self.attention_dropout1(attnention_output2, training=training)
        attnention_output2 = self.attention_normalization1(attnention_output1 + attnention_output2)

        output = self.feed_forward_layer(attnention_output2)
        output = self.feed_forward_dropout(output, training=training)
        output = self.feed_forward_normalization(attnention_output2 + output)

        return output, attnention_weights1, attnention_weights2


class Encoder(Layer):
    def __init__(self, num_neurons, num_hidden_neurons, num_heads, vocabular_size, num_enc_layers = 6):
        super(Encoder, self).__init__()
        
        self.num_enc_layers = num_enc_layers
        
        self.pre_processing_layer = PreProcessingLayer(num_neurons, vocabular_size)
        self.encoder_layers = [EncoderLayer(num_neurons, num_hidden_neurons, num_heads) for _ in range(num_enc_layers)]

    def call(self, sequence, training, mask):
        
        sequence = self.pre_processing_layer(sequence, training, mask)
        for i in range(self.num_enc_layers):
            sequence = self.encoder_layers[i](sequence, training, mask)

        return sequence


class Decoder(Layer):
    def __init__(self, num_neurons, num_hidden_neurons, num_heads, vocabular_size, num_dec_layers=6):
        super(Decoder, self).__init__()

        self.num_dec_layers = num_dec_layers
        
        self.pre_processing_layer = PreProcessingLayer(num_neurons, vocabular_size)
        self.decoder_layers = [DecoderLayer(num_neurons, num_hidden_neurons, num_heads) for _ in range(num_dec_layers)]

    def call(self, sequence, enconder_output, training, look_ahead_mask, padding_mask):

        sequence = self.pre_processing_layer(sequence, training, padding_mask)
        attention_weights = {}
        for i in range(self.num_dec_layers):

            sequence, attention_weights1, attention_weights2 = self.decoder_layers[i](sequence, enconder_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_attention_weights1'.format(i+1)] = attention_weights1
            attention_weights['decoder_layer{}_attention_weights2'.format(i+1)] = attention_weights2

        return sequence, attention_weights


#Transformer

class Transformer(Model):
    def __init__(self, num_layers, num_neurons, num_hidden_neurons, num_heads, input_vocabular_size, target_vocabular_size):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_neurons, num_hidden_neurons, num_heads, input_vocabular_size, num_layers)
        self.decoder = Decoder(num_neurons, num_hidden_neurons, num_heads, target_vocabular_size, num_layers)
        self.linear_layer = Dense(target_vocabular_size)

    def call(self, transformer_input, tar, training, encoder_padding_mask, look_ahead_mask, decoder_padding_mask):
        encoder_output = self.encoder(transformer_input, training, encoder_padding_mask)
        decoder_output, attention_weights = self.decoder(tar, encoder_output, training, look_ahead_mask, decoder_padding_mask)
        output = self.linear_layer(decoder_output)

        return output, attention_weights



#Training

#Learning Rate
class Schedule(LearningRateSchedule):
    def __init__(self, num_neurons, warmup_steps=4000):
        super(Schedule, self).__init__()

        self.num_neurons = tf.cast(num_neurons, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.num_neurons) * tf.math.minimum(arg1, arg2)


#Loss function

loss_objective_function = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def padded_loss_function(real, prediction):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = loss_objective_function(real, prediction)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_mean(loss)

training_loss = Mean(name='training_loss')
training_accuracy = SparseCategoricalAccuracy(name='training_accuracy')

# Initialize helpers
#data_container = DataHandler()
maskHandler = MaskHandler()

# Initialize parameters
num_layers = 4
num_neurons = 128 #FFN vom attention layer
num_hidden_layers = 512 #ffn
num_heads = 8 #Wie viele Heads

# Initialize vocabular size
input_vocablar_size = data_container.tokenizer_ru.vocab_size + 2
target_vocablar_size = data_container.tokenizer_en.vocab_size + 2

# Initialize learning rate
learning_rate = Schedule(num_neurons)
optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Initialize transformer
transformer = Transformer(num_layers, num_neurons, num_hidden_layers, num_heads, input_vocablar_size, target_vocablar_size)



#Richtiges Training

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(input_language, target_language):
    target_input = target_language[:, :-1]
    tartet_output = target_language[:, 1:]
    
    # Create masks
    encoder_padding_mask = maskHandler.padding_mask(input_language)
    decoder_padding_mask = maskHandler.padding_mask(input_language)


    look_ahead_mask = maskHandler.look_ahead_mask(tf.shape(target_language)[1])
    decoder_target_padding_mask = maskHandler.padding_mask(target_language)
    combined_mask = tf.maximum(decoder_target_padding_mask, look_ahead_mask)
    
    # Run training step
    with tf.GradientTape() as tape:
        predictions, _ = transformer(input_language, target_input,  True, encoder_padding_mask, combined_mask, decoder_padding_mask)
        total_loss = padded_loss_function(tartet_output, predictions)


    gradients = tape.gradient(total_loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    training_loss(total_loss)
    training_accuracy(tartet_output, predictions)


for epoch in tqdm(range(20)):
    training_loss.reset_states()
    training_accuracy.reset_states()

    for (batch, (input_language, target_language)) in enumerate(data_container.train_data):
        train_step(input_language, target_language)
    
    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch, train_loss.result(), train_accuracy.result()))