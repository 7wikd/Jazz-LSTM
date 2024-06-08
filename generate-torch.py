'''
generate-torch.py
1. The generate() function reads the notes from the notes file using the pickle module.
2. The sequences() function creates sequences of notes from the notes extracted in the generate() function. It then converts the notes to integers and normalizes them.
3. The generate_notes() function generates notes using the model, sequences, pitchnames, and n_vocab.
4. The create_midi() function creates a MIDI file using the notes generated in the generate_notes() function.
5. The if __name__ == '__main__': block calls the generate() function when the script is run.
'''
import pickle
import numpy as np
import torch
import torch.nn as nn
from music21 import instrument, note, stream, chord

class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MusicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, dropout=0.4, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.batch_norm(x[:, -1, :])
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

def generate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('misc/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))
    net_in, _ = sequences(notes, pitchnames, n_vocab)
    model = MusicLSTM(1, 512, n_vocab).to(device)
    model.load_state_dict(torch.load('./model_first.pth'))
    model.eval()
    prediction_output = generate_notes(model, net_in, pitchnames, n_vocab, device)
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
    net_out = np.eye(n_vocab)[net_out]

    return torch.tensor(net_in).float(), torch.tensor(net_out)

def generate_notes(model, net_in, pitchnames, n_vocab, device):
    start = np.random.randint(0, len(net_in) - 1)
    int_to_note = {num: note for num, note in enumerate(pitchnames)}
    pattern = net_in[start].unsqueeze(0).to(device)
    prediction_output = []

    for note_index in range(500):
        prediction_input = pattern / float(n_vocab)
        prediction = model(prediction_input)
        index = torch.argmax(prediction, dim=1).item()
        result = int_to_note[index]
        prediction_output.append(result)
        pattern = torch.cat((pattern[:, 1:, :], torch.tensor([[[index]]]).float().to(device)), dim=1)

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
