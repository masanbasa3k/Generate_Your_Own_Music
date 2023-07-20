import midi
import os
import util
import numpy as np

patterns = {}
dirs = ["archive/split_midi/split_midi"]
all_samples = []
all_lens = []
print ("Loading Songs...")
i = 0
for dir in dirs:
	for root, subdirs, files in os.walk(dir):
		for file in files:
			print(i)
			i+=1
			path = root + "/" + file
			if not (path.endswith('.mid') or path.endswith('.midi')):
				continue
			try:
				samples = midi.midi_to_samples(path)
			except:
				print ("ERROR ", path)
				continue
			if len(samples) < 8:
				continue
				
			samples, lens = util.generate_add_centered_transpose(samples)
			all_samples += samples
			all_lens += lens
	
assert(sum(all_lens) == len(all_samples))
print ("Saving " + str(len(all_samples)) + " samples...")
all_samples = np.array(all_samples, dtype=np.uint8)
all_lens = np.array(all_lens, dtype=np.uint32)
np.save('samples.npy', all_samples)
np.save('lengths.npy', all_lens)
print ("Done")