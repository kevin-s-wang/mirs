import sys
sys.path.append('../mirs')

import pandas as pd
from mirs.ai.models import clip_model as cm

def remove_quotes_and_duplicate_period(sentence):
  """Removes quotes from the start and end of a sentence and removes duplicate periods at the end.

  Args:
    sentence: The sentence to process.

  Returns:
    The sentence with quotes and duplicate periods removed.
  """

  if sentence.startswith('"') and sentence.endswith('"'):
    sentence = sentence[1:-1]

  # Remove duplicate periods at the end
  while sentence.endswith(".."):
    sentence = sentence[:-1]

  return sentence

def generate_caption_embeddings(caption: str):
    caption_embeddings = cm.get_text_embeddings([caption])
    return caption_embeddings.cpu().detach().numpy()[0]

if __name__ == '__main__':
    df1 = pd.read_parquet('.cache/prepare/data', engine='pyarrow')
    df2 = pd.read_parquet('.cache/prepare/augmented', engine='pyarrow')

    df1['caption'] = df1['caption'].apply(remove_quotes_and_duplicate_period)
    df1['caption_embeddings'] = df1['caption'].apply(generate_caption_embeddings)

    df2['caption'] = df2['caption'].apply(remove_quotes_and_duplicate_period)
    df2['caption_embeddings'] = df2['caption'].apply(generate_caption_embeddings)

    pd.concat([df1, df2], ignore_index=True) \
        .to_parquet('.cache/prepare/merged', 
                    engine='pyarrow', 
                    partition_cols=['dataset_name', 'split'], 
                    index=False)




