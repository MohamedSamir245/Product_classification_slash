import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pathlib import Path

import matplotlib.cm as cm
import seaborn as sns

sns.set_style('darkgrid')


from functions import get_paths_and_labels


BATCH_SIZE = 32
IMAGE_SIZE = (224,224)


train_root = Path('./data/external/train')
test_root = Path('./data/external/test')
valid_root = Path('./data/external/valid')

train_df= get_paths_and_labels(train_root)
valid_df= get_paths_and_labels(valid_root)
test_df= get_paths_and_labels(test_root)

train_label_counts= train_df['label'].value_counts()
train_label_counts_df=pd.DataFrame(train_label_counts)

fig=plt.figure(figsize=(20, 6))
sns.barplot(x=train_label_counts.index, y=train_label_counts.values,
            alpha=0.8, palette=sns.color_palette("YlOrRd_r", 40))
plt.title('Distribution of Labels in Image Dataset', fontsize=16)
plt.xlabel('Label', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
fig.savefig('./reports/figures/label_distribution.png')


random_index = np.random.randint(0, len(train_df), 16)
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16),
                         subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(train_df['image_path'][random_index[i]]))
    ax.set_title(train_df.label[random_index[i]], color="blue", fontsize=20)
plt.tight_layout()
fig.savefig('./reports/figures/random_images.png')