import pandas as pd
from tqdm import tqdm

import nltk
nltk.download('punkt')

original_words = set()
augmented_words = set()

data_dir = '/Users/I561210/Learning/mirs/.cache/prepare/merged'

data_df = pd.read_parquet(data_dir, engine='pyarrow', filters=[('split', '=', 'train')])

# tokenize stentence, remove punctuations, qoutes and convert to lower case
print('processing words...')
original_df = data_df[data_df['augmented'] == False]
original_df['caption'].apply(lambda x: original_words.update(nltk.word_tokenize(x)))

augmented_data_df = data_df[data_df['augmented'] == True]
augmented_data_df['caption'].apply(lambda x: augmented_words.update(nltk.word_tokenize(x)))


print('origional words: ', len(original_words))
print('augmented words: ', len(augmented_words))

count = 0

for word in augmented_words:
    if word not in original_words:
        count += 1

print('new words: ', count)



