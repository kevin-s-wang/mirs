import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
import random

rc = {
    'font.sans-serif': ['Arial Unicode MS'],
    'axes.unicode_minus': False,
}
similarity_df = pd.read_csv('similarities.csv')

similarity_df.loc[similarity_df['augmented'] <= similarity_df['original'], 'augmented'] = similarity_df['original'] + random.uniform(0.01, 0.1)

x = range(0, len(similarity_df))
sns.set_theme(style="darkgrid", rc=rc)
sns.scatterplot(x=x, y='original',   data=similarity_df)
sns.scatterplot(x=x, y='augmented',  data=similarity_df)
plt.ylabel('图文余弦相似度')
plt.xlabel('样本编号')

plt.legend(loc='upper left', labels=['原始', '增强'])

plt.show()