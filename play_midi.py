import pygame
import time

def load_midi(file_path):
    pygame.mixer.music.load(file_path)

def play_midi():
    pygame.mixer.music.play()

    # Add some delay to allow the MIDI to play (change the duration as needed)
    time.sleep(10)  # Play for 10 seconds, you can adjust the duration
    pygame.mixer.music.stop()

if __name__ == "__main__":
    pygame.init()

    midi_file_path = "gan_final.mid"
    load_midi(midi_file_path)
    play_midi()
