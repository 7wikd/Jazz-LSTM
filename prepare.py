'''
Preparing the model 
'''

import glob
import pickle
import numpy as np

from music21 import converter, instrument, note, chord

from keras.layers import Activation, BatchNormalization, Dense, Dropout, LSTM
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train():
    notes = get_notes()

    n_vocab = len(set(notes))

    net_input, net_output = sequence(notes, n_vocab)

    model = create_net(net_input, n_vocab)

    net(model, net_input, net_output)

def get_notes():
    notes = []
    for file in glob.glob('midi/*.mid'):
        midi = converter.parse(file)
        print("Parsing: %s " %file)

        notes_to_parse = None

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        
        for element in notes_to_parse:
            if isinstance(element,note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    
    with open('misc/notes','wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def sequence(notes, n_vocab):
    seq_len = 100
    
    pitchnames = sorted(set(item for item in notes))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    net_in = []
    net_out = []

    for i in range(0,len(notes)-seq_len,1):
        seq_in = notes[i: i+seq_len]
        seq_out = notes[i+seq_len]
        
        net_in.append([note_to_int[char] for char in seq_in])
        net_out.append(note_to_int[seq_out])
    
    n_patterns = len(net_in)

    net_in = np.reshape(net_in,(n_patterns,seq_len,1))
    net_in = net_in /float(n_vocab)

    net_out = np_utils.to_categorical(net_out)

    return (net_in,net_out)

def create_net(net_in, n_vocab):
    model = Sequential([
        LSTM(
            512,
            input_shape = (net_in.shape[1],net_in.shape[2]),
            recurrent_dropout=0.4,
            return_sequences=True
        ),
        LSTM(
            512, 
            return_sequences=True, 
            recurrent_dropout=0.4
        ),
        LSTM(512),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256),
        Activation('relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(n_vocab),
        Activation('softmax')
    ])
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.01))
    
    return model

def net(model, net_in,net_out):
    filepath = 'model_first.hdf5'
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model.fit(net_in,net_out,epochs=200,batch_size=64,callbacks=callbacks_list)
  
if __name__ == '__main__':
    train()