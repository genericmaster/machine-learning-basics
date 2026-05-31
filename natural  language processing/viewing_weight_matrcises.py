import numpy as np
import pandas as pd

# 1. Load the binary files
# Ensure the paths match where your training script saved them
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

words = np.load(os.path.join(BASE_DIR, 'vocab.npy'), allow_pickle=True)
all_weights = np.load(os.path.join(BASE_DIR, 'word_vectors.npy'), allow_pickle=True)

# 2. Extract the weights matrix
# model.get_weights() returns a list: [weights_matrix, bias_vector]
# We need index [0] for the 93x100 matrix
embeddings = all_weights[()]

# 3. Verify shapes match before creating DataFrame
print(f"Matrix shape: {embeddings.shape}") # Should be (93, 100)
print(f"Vocab size: {len(words)}")        # Should be 93

# 4. Create the readable table
# The 93 words become the row labels for the 100-column matrix
df = pd.DataFrame(embeddings, index=words)

# 5. Export to CSV
df.to_csv('my_word_embeddings.csv')

print("Success! 'my_word_embeddings.csv' has been created.")

print(df)