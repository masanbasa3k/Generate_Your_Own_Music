import mido

def is_valid_message(msg):
    # Filter out MetaMessage and other non-relevant message types
    return not isinstance(msg, mido.MetaMessage)

def message_to_string(msg):
    # Convert the Message object to a string representation
    return str(msg)

def create_vocabulary(midi_file_path):
    # Load and parse the MIDI file
    mid = mido.MidiFile(midi_file_path)

    # Initialize an empty set to store unique MIDI messages
    unique_messages = set()

    # Iterate through all the MIDI messages and extract unique relevant messages
    for track in mid.tracks:
        for msg in track:
            if is_valid_message(msg):
                msg_str = message_to_string(msg)
                unique_messages.add(msg_str)

    # Create a dictionary to store the vocabulary with MIDI messages as keys and integer IDs as values
    vocabulary = {msg: i for i, msg in enumerate(unique_messages)}

    return vocabulary

def message_to_int(msg, vocabulary):
    # Convert the Message object to its corresponding integer ID using the vocabulary
    return vocabulary[str(msg)]

def preprocess_midi_data(midi_file_path, vocabulary):
    # Load and parse the MIDI file
    mid = mido.MidiFile(midi_file_path)

    # Initialize a list to store the numerical sequences
    numerical_sequences = []

    # Iterate through all the MIDI messages and extract relevant messages
    current_sequence = []
    for track in mid.tracks:
        for msg in track:
            if is_valid_message(msg):
                current_sequence.append(message_to_int(msg, vocabulary))

        # Add the current sequence to the list of numerical sequences
        numerical_sequences.append(current_sequence)
        current_sequence = []  # Reset the sequence for the next track

    return numerical_sequences


def midi_to_numeric(path):
    midi_file_path = path
    vocabulary = create_vocabulary(midi_file_path)
    numerical_sequences = preprocess_midi_data(midi_file_path, vocabulary)
    return numerical_sequences

if '__name__' == '__main__':
    midi_to_numeric('archive/1.mid')