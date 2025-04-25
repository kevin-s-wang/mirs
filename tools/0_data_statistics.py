import pandas as pd
from tqdm import tqdm

data_dir = '/Users/I561210/Learning/mirs/.cache/prepare/merged'

data_df = pd.read_parquet(data_dir, engine='pyarrow', filters=[('split', '=', 'train')])
                
augmented_data_df = data_df[data_df['augmented'] == True]

# print(augmented_data_df.groupby('dataset_name', observed=True)['filepath'].nunique())

augmented_data_df = augmented_data_df.drop_duplicates(subset=['filepath'])
print(augmented_data_df.groupby('dataset_name', observed=True)['filepath'].count())
# data_df = data_df[data_df['filepath'].isin(augmented_data_df['filepath'])]


augmented_similarities = augmented_data_df['cosine_similarity'].values
augmented_filepaths = augmented_data_df['filepath'].values

original_similarities = []
for i, augmented_filepath in tqdm(enumerate(augmented_filepaths)):
    similarities = data_df.loc[data_df['filepath'] == augmented_filepath, 'cosine_similarity']
    original_similarities.append(similarities.mean())


print('Original similarities: ', len(original_similarities))

original_similarities = pd.Series(original_similarities)

similarity_df = pd.DataFrame({
    'original': original_similarities, 
    'augmented': augmented_similarities})

similarity_df.to_csv('similarities.csv', index=True)

print('plotting...')
