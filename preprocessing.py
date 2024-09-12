import numpy as np
import string
import re

class PreprocessingUnit:

    def __init__(self, file, enc='utf-8'):

        self.text = file
        with open(self.text, 'rt', encoding=enc) as file:
            self.text = file.read()
        self.encoding = enc
        self.vocab_size = 0
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.text_as_int = np.empty(1)
        self.inputs = None
        self.targets = None 
        self.tokens = None

    def lower_case(self):
        
        self.text = self.text.lower()
        print("All characters are lowercase")

    def upper_case(self):

        self.text = self.text.upper()
        print("All characters are upper case")

    def remove_punctuation(self):

        self.text = self.text.translate(str.maketrans("", "", string.punctuation))
        print("All punctuation has been removed")

    def remove_newlines(self):

        self.text = self.text.replace('\n', ' ').replace('\r', '')
        print("All newlines have been removed")

    def build_vocab(self):

        # Create a list of unique characters in the text
        chars = sorted(list(set(self.text)))
        self.vocab_size = len(chars)

        # Create character to integer and integer to character mappings
        self.token_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_token = {idx: char for idx, char in enumerate(chars)}

        # Convert the entire text into a list of integer indices
        self.text_as_int = np.array([self.token_to_idx[char] for char in self.text])

        print("Text has been translated to integers")
        print(f"Corpus length: {len(self.text)}")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def create_batches(self, seq_length, batch_size):
        """
        Create batches of sequences and corresponding targets
        """
    
        # Calculate how many full batches we can make from the dataset
        n_batches = len(self.text_as_int) // (batch_size * seq_length)
        # Trim the text to fit exactly into full batches
        text_trimmed = self.text_as_int[:n_batches * batch_size * seq_length + 1]
        
        # Initialize inputs and targets arrays
        self.inputs = np.zeros((n_batches, batch_size, seq_length), dtype=int)
        self.targets = np.zeros((n_batches, batch_size, seq_length), dtype=int)

        # Loop through each batch and fill in the input and target sequences
        for i in range(n_batches):
            for j in range(batch_size):
                # Find starting index for the batch sequence
                start_idx = i * batch_size * seq_length + j * seq_length

                # Fill input sequence 
                self.inputs[i, j, :] = text_trimmed[start_idx:start_idx + seq_length]

                # Fill target sequence 
                self.targets[i, j, :] = text_trimmed[start_idx + 1:start_idx + seq_length + 1]

        print("Batch creation successful")
        print(f"Batches created: {n_batches}")