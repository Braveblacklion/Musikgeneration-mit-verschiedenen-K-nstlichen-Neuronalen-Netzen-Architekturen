###################################################################################################
#####  Quelle: https://github.com/davidADSP/GDL_code/blob/master/models/RNNAttention.py ###########
###################################################################################################

import numpy as np
#from keras.utils.np_utils import to_categorical
from keras.utils import np_utils

from tensorflow.keras.layers import LSTM, Input, Dropout, Dense, Activation, Embedding, Concatenate, Reshape
from tensorflow.keras.layers import Flatten, RepeatVector, Permute, TimeDistributed
from tensorflow.keras.layers import Multiply, Lambda, Softmax
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


def get_distinct(elements):
    # Get all pitch names
    element_names = sorted(set(elements))
    n_elements = len(element_names)
    return (element_names, n_elements)


def create_lookups(element_names):
    # create dictionary to map notes and durations to integers
    element_to_int = dict((element, number) for number, element in enumerate(element_names))
    int_to_element = dict((number, element) for number, element in enumerate(element_names))

    return (element_to_int, int_to_element)

def prepare_sequences(commands, values1, durations, lookups, distincts, seq_len=32):
    """ Prepare the sequences used to train the Neural Network """

    command_to_int, int_to_command, value1_to_int, int_to_value1, duration_to_int, int_to_duration = lookups
    command_names, n_commands, value1_names, n_value1, duration_names, n_durations = distincts

    commands_network_input = []
    commands_network_output = []
    values1_network_input = []
    values1_network_output = []
    durations_network_input = []
    durations_network_output = []

    # create input sequences and the corresponding outputs
    for i in range(len(commands) - seq_len):
        commands_sequence_in = commands[i:i + seq_len]
        commands_sequence_out = commands[i + seq_len]
        commands_network_input.append([command_to_int[char] for char in commands_sequence_in])
        commands_network_output.append(command_to_int[commands_sequence_out])

        values1_sequence_in = values1[i:i + seq_len]
        values1_sequence_out = values1[i + seq_len]
        values1_network_input.append([value1_to_int[char] for char in values1_sequence_in])
        values1_network_output.append(value1_to_int[values1_sequence_out])

        durations_sequence_in = durations[i:i + seq_len]
        durations_sequence_out = durations[i + seq_len]
        durations_network_input.append([duration_to_int[char] for char in durations_sequence_in])
        durations_network_output.append(duration_to_int[durations_sequence_out])

    n_patterns = len(commands_network_input)

    # reshape the input into a format compatible with LSTM layers
    commands_network_input = np.reshape(commands_network_input, (n_patterns, seq_len))
    values1_network_input = np.reshape(values1_network_input, (n_patterns, seq_len))
    durations_network_input = np.reshape(durations_network_input, (n_patterns, seq_len))


    network_input = [commands_network_input, values1_network_input, durations_network_input]

    commands_network_output = np_utils.to_categorical(commands_network_output, num_classes=n_commands)
    values1_network_output = np_utils.to_categorical(values1_network_output, num_classes=n_value1)
    durations_network_output = np_utils.to_categorical(durations_network_output, num_classes=n_durations)
    network_output = [commands_network_output, values1_network_output, durations_network_output]

    return (network_input, network_output)

def create_network(n_commands, n_value1, n_durations, embed_size=100, rnn_units=256, use_attention=False):
    """ create the structure of the neural network """

    commands_in = Input(shape=(None,), name="commands_channels_in")
    value1_in = Input(shape=(None,), name="values1_in")
    durations_in = Input(shape=(None,), name="durations_in")

    x1 = Embedding(n_commands, embed_size)(commands_in)
    x2 = Embedding(n_value1, embed_size)(value1_in)
    x3 = Embedding(n_durations, embed_size)(durations_in)

    x = Concatenate()([x1, x2, x3])

    x = LSTM(rnn_units, return_sequences=True)(x)
    #x = Dropout(0.2)(x)

    if use_attention:

        x = LSTM(rnn_units, return_sequences=True)(x)
        #x = Dropout(0.2)(x)

        e = Dense(1, activation='tanh')(x)
        e = Reshape([-1])(e)
        alpha = Activation('softmax')(e)

        alpha_repeated = Permute([2, 1])(RepeatVector(rnn_units)(alpha))

        c = Multiply()([x, alpha_repeated])
        c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(rnn_units,))(c)

    else:
        c = LSTM(rnn_units)(x)
        #c = Dropout(0.2)(c)

    commands_out = Dense(n_commands, activation='softmax', name='commands_channels_out')(c)
    value1_out = Dense(n_value1, activation='softmax', name='values1_out')(c)
    durations_out = Dense(n_durations, activation='softmax', name='durations_out')(c)

    model = Model([commands_in, value1_in, durations_in], [commands_out, value1_out, durations_out])

    if use_attention:
        att_model = Model([commands_in, value1_in, durations_in], alpha)
    else:
        att_model = None

    opti = RMSprop(lr=0.005)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], optimizer=opti)

    return model, att_model

def sample_with_temp(preds, temperature):
    if temperature == 0:
        return np.argmax(preds)
    else:
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)
