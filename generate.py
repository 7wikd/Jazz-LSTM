'''
generate.py
1. The generate() function reads the notes from the notes file using the pickle module.
2. The sequences() function creates sequences of notes from the notes extracted in the generate() function. It then converts the notes to integers and normalizes them.
3. The create_net() function creates a neural network model using the Keras Sequential API. The model consists of three LSTM layers, two Dropout layers, two BatchNormalization layers, and two Dense layers.
4. The generate_notes() function generates notes using the model, sequences, pitchnames, and n_vocab.
5. The create_midi() function creates a MIDI file using the notes generated in the generate_notes() function.
6. The if __name__ == '__main__': block calls the generate() function when the script is run.
'''
import pickle
import numpy as np

from music21 import instrument, note, stream, chord

from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Dense, Dropout, LSTM
from keras.utils import to_categorical

def generate():
    with open('misc/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))
    net_in, _ = sequences(notes, pitchnames, n_vocab)
    model = create_net(net_in, n_vocab)
    model.load_weights('./model_first.keras')
    prediction_output = generate_notes(model, net_in, pitchnames, n_vocab)
    create_midi(prediction_output)

def sequences(notes, pitchnames, n_vocab):
    seq_len = 100
    note_to_int = {note: num for num, note in enumerate(pitchnames)}
    net_in = []
    net_out = []

    for i in range(0, len(notes) - seq_len, 1):
        in_seq = notes[i:i + seq_len]
        out_seq = notes[i + seq_len]
        net_in.append([note_to_int[char] for char in in_seq])
        net_out.append(note_to_int[out_seq])

    n_patterns = len(net_in)
    net_in = np.reshape(net_in, (n_patterns, seq_len, 1))
    net_in = net_in / float(n_vocab)
    net_out = to_categorical(net_out)

    return net_in, net_out

def create_net(net_in, n_vocab):
    model = Sequential([
        LSTM(512, input_shape=(net_in.shape[1], net_in.shape[2]), recurrent_dropout=0.4, return_sequences=True),
        LSTM(512, return_sequences=True, recurrent_dropout=0.4),
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
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def generate_notes(model, net_in, pitchnames, n_vocab):
    start = np.random.randint(0, len(net_in) - 1)
    int_to_note = {num: note for num, note in enumerate(pitchnames)}
    pattern = net_in[start]
    prediction_output = []

    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = [note.Note(int(current_note)) for current_note in notes_in_chord]
            for new_note in notes:
                new_note.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output.mid')

if __name__ == '__main__':
    generate()
