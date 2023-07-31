import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from music21 import stream, note, chord, instrument
from create_generator_model import get_notes, create_midi, LATENT_DIMENSION

def generate_music(generator_model, latent_dim, n_vocab, length=500):
    """ Generate new music using the trained generator model """
    # Create random noise as input to the generator
    noise = np.random.normal(0, 1, (1, latent_dim))
    predictions = generator_model.predict(noise)
    
    # Scale back the predictions to the original range
    pred_notes = [x * (n_vocab / 2) + (n_vocab / 2) for x in predictions[0]]
    
    # Map generated integer indices to note names
    pitchnames = sorted(set(item for item in notes))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    pred_notes_mapped = [int_to_note[int(x)] for x in pred_notes]
    
    return pred_notes_mapped[:length]

if __name__ == '__main__':
    # Load the trained generator model
    generator_model = load_model("generator_model.h5")
    
    # Load the processed notes and get the number of unique pitches
    notes = get_notes()
    n_vocab = len(set(notes))
    
    # Generate new music sequence
    generated_music = generate_music(generator_model, LATENT_DIMENSION, n_vocab)
    
    # Create a MIDI file from the generated music
    create_midi(generated_music, 'generated_music')
