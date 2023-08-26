import pygame
import time


def play_midi(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(30) 

if __name__ == "__main__":
    pygame.init()

    midi_file_path = "generated_music.mid"
    play_midi(midi_file_path)
