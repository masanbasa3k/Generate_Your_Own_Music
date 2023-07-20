import numpy as np

npy_file = "output_combined.npy" 

try:
    midi_data = np.load(npy_file, allow_pickle=True)
    print("Loaded MIDI data:")
    print(midi_data)
    print("Data type:", type(midi_data))
    print("Data shape:", midi_data.shape)
    if isinstance(midi_data, list):
        print("First item type:", type(midi_data[0]))
    elif isinstance(midi_data, np.ndarray):
        print("Data dtype:", midi_data.dtype)
        print("First item shape:", midi_data[0].shape)
    else:
        print("Unknown data type")
except Exception as e:
    print("Error loading the npy file:", e)