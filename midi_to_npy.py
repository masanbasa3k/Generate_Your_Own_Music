import mido
import numpy as np
import os

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

def create_combined_npy(input_directory, output_npy):
    midi_files = [f for f in os.listdir(input_directory) if f.endswith('.mid')]

    # Store all MIDI events from different files as a list of tuples
    all_midi_events = []
    for midi_file in midi_files:
        midi_path = os.path.join(input_directory, midi_file)

        midi_data = mido.MidiFile(midi_path)
        for track in midi_data.tracks:
            for msg in track:
                if msg.type in {'note_on', 'note_off'}:
                    event = (
                        msg.type,
                        msg.time,
                        msg.note,
                        msg.velocity
                    )
                    all_midi_events.append(event)

    # Convert the list of tuples to a structured NumPy array
    midi_np_array = np.array(all_midi_events, dtype=[
        ('type', 'U8'),        # Unicode string, maximum 8 characters
        ('time', 'int'),       # Integer for time
        ('note', 'int'),       # Integer for MIDI note number
        ('velocity', 'int')    # Integer for velocity
    ])

    np.save(output_npy, midi_np_array)

if __name__ == "__main__":
    input_directory = "archive"  # Replace with the path to your directory containing MIDI files
    output_npy = "output_combined.npy"         # Replace with the desired output npy file path

    create_combined_npy(input_directory, output_npy)
