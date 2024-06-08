'''
prepare-torch.py
1. The train() function reads the notes from the notes file using the pickle module.
2. The get_notes() function reads all the MIDI files in the midi directory and extracts the notes and chords from them. It then saves the notes in a file named notes using the pickle module.
3. The sequence() function creates sequences of notes from the notes extracted in the get_notes() function. It then converts the notes to integers and normalizes them.
4. The MusicLSTM class defines the neural network model using the PyTorch nn.Module class. The model consists of an LSTM layer, a BatchNorm1d layer, and two Dense layers.
5. The MusicDataset class defines a custom dataset class using the PyTorch Dataset class.
6. The train() function calls the get_notes() function to extract the notes from the MIDI files, creates sequences of notes using the sequence() function, creates a neural network model using the MusicLSTM class, and trains
the model using the DataLoader class.
7. The if __name__ == '__main__': block calls the train() function when the script is run.
'''

import glob
import pickle
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from music21 import converter, instrument, note, chord

# Define the LSTM model
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

# Custom dataset class
class MusicDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    notes = get_notes()
    n_vocab = len(set(notes))
    net_input, net_output = sequence(notes, n_vocab)
    dataset = MusicDataset(net_input, net_output)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = MusicLSTM(1, 512, n_vocab).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device).argmax(dim=1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")
        torch.save(model.state_dict(), 'model_first.pth')

def get_notes():
    notes = []
    for file in glob.glob('midi/*.mid'):
        midi = converter.parse(file)
        print(f"Parsing: {file}")
        notes_to_parse = None

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('misc/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def sequence(notes, n_vocab):
    seq_len = 100
    pitchnames = sorted(set(item for item in notes))
    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    net_in = []
    net_out = []

    for i in range(0, len(notes) - seq_len, 1):
        seq_in = notes[i: i + seq_len]
        seq_out = notes[i + seq_len]
        net_in.append([note_to_int[char] for char in seq_in])
        net_out.append(note_to_int[seq_out])

    n_patterns = len(net_in)
    net_in = np.reshape(net_in, (n_patterns, seq_len, 1))
    net_in = net_in / float(n_vocab)
    net_out = np.eye(n_vocab)[net_out]

    return torch.tensor(net_in), torch.tensor(net_out)

if __name__ == '__main__':
    train()
