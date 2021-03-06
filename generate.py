'''
Generating music 
'''

import pickle
import numpy as np

from music21 import instrument,note, stream, chord

from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Dense, Dropout, LSTM

def generate():
    with open('misc/notes','rb') as filepath:
        notes = pickle.load(filepath)

        pitchnames = sorted(set(item for item in notes))
        n_vocab = len(set(notes))

        net_in,norm_out = sequences(notes, pitchnames, n_vocab)
        model = net(norm_in,n_vocab)
        pred_op = generate_notes(model, net_in, pitchnames, n_vocab)
        create_midi(pred_op)

def sequences(notes,pitchnames,n_vocab):
