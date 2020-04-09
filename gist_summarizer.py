from keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import nltk
from attention import AttentionLayer
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import tensorflow as tf
import pickle
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

with open('trained_model/contraction_mapping.pickle', 'rb') as handle:
    contraction_mapping = pickle.load(handle)


nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def text_cleaner(text, num):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"', '', newString)
    newString = ' '.join(
        [contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
    newString = re.sub(r"'s\b", "", newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    newString = re.sub('[m]{2,}', 'mm', newString)
    if(num == 0):
        tokens = [w for w in newString.split() if not w in stop_words]
    else:
        tokens = newString.split()
    long_words = []
    for i in tokens:
        if len(i) > 1:  # removing short word
            long_words.append(i)
    return (" ".join(long_words)).strip()



max_text_len = 30
max_summary_len = 8


tempjson = open("trained_model/x_tokenizer.json", "r").read()
x_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tempjson)

tempjson2 = open("trained_model/y_tokenizer.json", "r").read()
y_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tempjson2)

tempjson2 = open("trained_model/y_tokenizer.json", "r").read()
y_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tempjson2)

latent_dim = 300
embedding_dim = 100

model = tf.keras.models.load_model('trained_model/gce-savedmodel')
model.summary()

reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index

encoder_model = tf.keras.models.load_model('trained_model/gce-encoder_model')
decoder_model = tf.keras.models.load_model('trained_model/gce-decoder_model')

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict(
            [target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if(sampled_token != 'eostok'):
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_summary_len - 1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

def seq2summary(input_seq):
    newString = ''
    for i in input_seq:
        if((i != 0 and i != target_word_index['sostok']) and i != target_word_index['eostok']):
            newString = newString + reverse_target_word_index[i] + ' '
    return newString

def seq2text(input_seq):
    newString = ''
    for i in input_seq:
        if(i != 0):
            newString = newString + reverse_source_word_index[i] + ' '
    return newString

def preprocess_text(stringIn):
    if type(stringIn) is list:
        cleanedString = []
        for t in stringIn:
            cleanedString.append(text_cleaner(t, 0))
        return cleanedString
    elif type(stringIn) is str:
        temp = []
        temp.append(str)
        preprocess_text(temp)
    else:
        return None

def summarise_from_clean_text(stringIn):
    tokenSequence = x_tokenizer.texts_to_sequences(stringIn) #create the input sequence of cleaned string
    paddedSequence = pad_sequences(tokenSequence,  maxlen=max_text_len, padding='post') #pad the seq to max len
    summary = decode_sequence(paddedSequence)
    return summary

print("TENSOR MODEL READY")
