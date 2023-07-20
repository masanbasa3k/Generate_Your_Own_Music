import mido
import numpy as np

def midi_to_numpy(midi_filename):
    midi_data = mido.MidiFile(midi_filename)

    # Store MIDI events as a list of dictionaries
    midi_events = []
    for track in midi_data.tracks:
        for msg in track:
            if msg.type in {'note_on', 'note_off'}:
                event = {
                    'type': msg.type,
                    'time': msg.time,
                    'note': msg.note,
                    'velocity': msg.velocity
                }
                midi_events.append(event)

    # Convert the list of dictionaries to a structured NumPy array
    midi_np_array = np.array(midi_events, dtype=[
        ('type', 'U8'),        # Unicode string, maximum 8 characters
        ('time', 'int'),       # Integer for time
        ('note', 'int'),       # Integer for MIDI note number
        ('velocity', 'int')    # Integer for velocity
    ])

    return midi_np_array

def save_npy_file(midi_np_array, npy_filename):
    np.save(npy_filename, midi_np_array)

if __name__ == "__main__":
    input_midi_file = "archive/split_midi/split_midi/song19_2348.mid"  # Replace with the path to your MIDI file
    output_npy_file = "song19_2345.npy"      # Replace with the desired output path

    midi_np_array = midi_to_numpy(input_midi_file)
    save_npy_file(midi_np_array, output_npy_file)
