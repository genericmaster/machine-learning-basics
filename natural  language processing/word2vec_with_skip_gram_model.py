
import numpy as np 
from sklearn.preprocessing import LabelBinarizer
import keras
import string
import os
import re
from keras.callbacks import LearningRateScheduler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from collections import Counter
import pdfplumber

def load_pdf_corpus(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text and len(page_text.strip()) > 50:
                text += page_text + " "
            if i % 100 == 0:
                print(f"  processed page {i}/{len(pdf.pages)}")
    
    # preprocessing outside the with block
    text = text.lower()
    text = text.replace("\u2019s", "")
    text = text.replace("'s", "")
    text = text.replace("\u2018", "")
    text = text.replace("\u2019", "")
    text = text.replace("\u201c", "")
    text = text.replace("\u201d", "")
    text = text.replace('\u2013', ' ')
    text = text.replace('\u2014', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)

    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'shall', 'can', 'it', 'its', 'this',
        'that', 'these', 'those', 'i', 'he', 'she', 'they', 'we', 'you', 'me',
        'him', 'her', 'them', 'us', 'my', 'his', 'your', 'our', 'their', 'what',
        'which', 'who', 'not', 'so', 'if', 'as', 'up', 'out', 'said', 'into',
        'then', 'than', 'there', 'when', 'all', 'just', 'more', 'also', 'about',
        'back', 'no', 'one', 'his', 'her', 'him',
        'know', 'looked', 'asked', 'looking', 'well', 'got', 'get', 'like',
        'now', 'still', 'even', 'though', 'very', 'much', 'over', 'after',
        'before', 'again', 'around', 'went', 'came', 'come', 'going',
        'think', 'thought', 'told', 'look', 'felt', 'feel', 'knew',
        'wanted', 'want', 'take', 'took', 'away', 'never', 'always',
        'every', 'here', 'see', 'seen', 'saw', 'made', 'make', 'way', 'too',
        'how', 'some', 'any', 'down', 'off', 'own', 'right', 'something',
        'nothing', 'anything', 'everything', 'put', 'let',
        'without', 'through', 'where', 'while', 'however', 'because', 'both',
        'oh', 'yes', 'yeah', 'okay', 'dont', 'cant', 'wont', 'wasnt',
        'isnt', 'arent', 'didnt', 'doesnt', 'havent', 'hasnt', 'wouldnt',
        'couldnt', 'shouldnt', 'im', 'ive', 'id', 'theyre', 'weve', 'youre',
        'hes', 'shes', 'thats', 'whats', 'youve', 'theyll', 'itll',
    }

    master_list = [word for word in text.split() if word not in stop_words and len(word) > 2]
    
    word_counts = Counter(master_list)
    master_list = [word for word in master_list if word_counts[word] >= 5]

    return master_list, word_counts
master_list, word_counts = load_pdf_corpus(r"C:\Users\rakhi\Downloads\harrypotter.pdf")
print(f"Total characters extracted: {len(master_list)}")


binazer = LabelBinarizer()

master_dictionary = binazer.fit(master_list)
vocab_size = len(binazer.classes_)

#unigram sampling distribution
word_counts = Counter(master_list)
count = [word_counts[word] for word in binazer.classes_]
unigram_array = np.array(count)
unigram_array = unigram_array ** (3/4)
unigram_array = unigram_array / unigram_array.sum()

words_array = np.array([np.searchsorted(binazer.classes_, w) for w in master_list])
window = 5

x_center, x_context, y = [], [], []

for offset in range(1, window + 1):
    # forward
    centers = words_array[:-offset]
    contexts = words_array[offset:]
    x_center.append(centers)
    x_context.append(contexts)
    y.append(np.ones(len(centers)))
    
    # negative samples
    for _ in range(5):
        fake = np.random.choice(len(binazer.classes_), size=len(centers), p=unigram_array)
        x_center.append(centers)
        x_context.append(fake)
        y.append(np.zeros(len(centers)))
    
    # backward
    centers = words_array[offset:]
    contexts = words_array[:-offset]
    x_center.append(centers)
    x_context.append(contexts)
    y.append(np.ones(len(centers)))
    
    for _ in range(5):
        fake = np.random.choice(len(binazer.classes_), size=len(centers), p=unigram_array)
        x_center.append(centers)
        x_context.append(fake)
        y.append(np.zeros(len(centers)))

x_train_center = np.concatenate(x_center)
x_train_context = np.concatenate(x_context)
y_train = np.concatenate(y)

print(f"Training pairs: {len(y_train)}")



def create_model():
   center_input = keras.Input(shape=(1,), name="center")
   context_input = keras.Input(shape=(1,), name="context")

   embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=128)

   center_embedding = embedding(center_input)
   context_embedding = embedding(context_input)

   center_embedding = keras.layers.Flatten()(center_embedding)
   context_embedding = keras.layers.Flatten()(context_embedding)

   dot_product = keras.layers.Dot(axes=1)([center_embedding, context_embedding])

   output = keras.layers.Dense(1, activation="sigmoid")(dot_product)

   model = keras.Model(inputs=[center_input, context_input], outputs=output)
   model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.016), loss="binary_crossentropy",)
        
   return model,embedding

def train_model(model, x_train_context, x_train_center, y_train, binazer, embedding):
    
    def lr_schedule(epoch, lr):
        initial_lr = 0.016
        final_lr = 0.0001
        total_epochs = 100
        decayed = initial_lr - (initial_lr - final_lr) * (epoch / total_epochs)
        return max(decayed, final_lr)
    callbacks = [
        LearningRateScheduler(lr_schedule, verbose=0)
    ]

    history = model.fit(
        x=[x_train_center, x_train_context],
        y=y_train,
        epochs=100,
        batch_size=8192,
        callbacks=callbacks
    )

    weights = embedding.get_weights()[0]
    np.save(os.path.join(BASE_DIR, 'word_vectors.npy'), weights)
    np.save(os.path.join(BASE_DIR, 'vocab.npy'), binazer.classes_)

    print("Files saved to:", BASE_DIR)
    print("Vocab size:", len(binazer.classes_))
    print("Weights shape:", weights.shape)

    return history
    
model_1, embedding = create_model()

output_model = train_model(model_1,x_train_context,x_train_center,y_train,binazer,embedding) 
print(output_model)


