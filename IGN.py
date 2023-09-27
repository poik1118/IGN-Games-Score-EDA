# DataSet Source : https://www.kaggle.com/datasets/joebeachcapital/ign-games?resource=download
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# %%
dataset = pd.read_csv('/workspaces/IGN-Games-Score-EDA/Dataset/ign.csv')

#%%
dataset.shape
#%%
dataset.head()
#%%
dataset.tail()
#%%
dataset.describe(include='all')
#%%
dataset.info
# %%
dataset.columns
# %%
dataset.iloc[:,1].value_counts()

# %%
plt.xticks(rotation=40)
plt.xlabel('score_phrase')
plt.bar(dataset.iloc[:,1].value_counts().index, dataset.iloc[:,1].value_counts())

# %%
dataset.platform.value_counts()
# %%
plt.figure(figsize=(40,15))
plt.xticks(rotation=60)
plt.xlabel('Platform')
plt.bar(dataset.platform.value_counts().index, dataset.platform.value_counts())

#%%
dataset.score.value_counts()
#%%
plt.xlabel('Score')
plt.ylabel('Count')
plt.figure(figsize=(20, 5))
plt.bar(dataset.score.value_counts().index, dataset.score.value_counts(), width=0.3, edgecolor='black', linewidth=0)
# %%
sns.histplot(dataset['score'], kde=True)    # 히스토그램 그래프, kde : 커널밀도추정으로 분포곡선 표현, 계단형태로 표현, 채우기 False
plt.title('Distribution of Game Scores')
plt.xlabel('Score')
plt.ylabel('Number of Games')
plt.show()

#%%
dataset.release_year.value_counts()
# %%
dataset['release_year'].value_counts().sort_index().plot(kind='barh', figsize=(12, 6))
plt.title('Number of Games Realeased Each Year')
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.show()

#%%
dataset.release_month.value_counts()
#%%
plt.title('Number of Games Realeased Each Month')
plt.xlabel('Month')
plt.ylabel('Number of Games')
plt.bar(dataset.release_month.value_counts().index, dataset.release_month.value_counts())

#%%
dataset.release_day.value_counts()
#%%
plt.title('Number of Games Realeased Each Day')
plt.xlabel('Day')
plt.ylabel('Number of Game')
plt.bar(dataset.release_day.value_counts().index, dataset.release_day.value_counts())

#%%
number_type_columns = dataset.select_dtypes('number')
number_type_columns.drop(['Unnamed: 0'], axis=1, inplace=True)
number_type_columns
#%%
sns.heatmap(number_type_columns.corr(), annot=True, vmin=-1, vmax=1)

# %%
filtered_dataset = dataset[dataset['release_year'] > 2008]

max_scores = filtered_dataset.groupby('release_year')['score'].max().reset_index()

highest_score_data = pd.merge(max_scores, filtered_dataset, on=['release_year', 'score'], how='left').drop_duplicates(subset=['release_year', 'score'])

fig, ax = plt.subplots(figsize=(25, 10))

ax.scatter(highest_score_data['release_year'], highest_score_data['score'], color='blue', s=100)
for x, y, title, genre in \
zip(highest_score_data['release_year'], highest_score_data['score'], highest_score_data['title'], highest_score_data['genre']):
    ax.annotate(f"{title} ({genre})", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

ax.set_title("Highest Scoring Game of Each Year (After 2008) with Genre")
ax.set_ylabel("Score")
ax.set_xlabel("Year")
ax.grid(True, which='both', linestyle='--', linewidth=1.0)
plt.tight_layout()
plt.show()

# %%
top_games_each_yer = dataset.loc[dataset.groupby('release_year')['score'].idxmax()]
print(top_games_each_yer[['release_year', 'title', 'score']])

#%%
dataset.genre.value_counts()
# %%
dataset['genre'].value_counts().plot(kind='bar', figsize=(20, 6))
plt.title('Number of Games Reviewed by Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Games')
plt.show()

#%%
dataset.editors_choice.value_counts()
# %%
plt.title('Distribution of Editor\'s Choice Games')
plt.xlabel('Number of Games')
plt.ylabel('Editor\'s Choice')
plt.barh(dataset.editors_choice.value_counts().index, dataset.editors_choice.value_counts(), color = ['red', 'green'], alpha=0.5)
plt.show()

#%%
dataset.platform.value_counts()
# %%
dataset.groupby('platform')['score'].mean().sort_values(ascending=False).plot(kind='bar', figsize=(12, 6))
plt.title('Average Scores by Platform')
plt.xlabel('Platform')
plt.ylabel('Average Score')
plt.show()

#%%
genre_avg_scores = dataset.groupby('genre')['score'].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(15, 7))

genre_avg_scores.plot(kind='bar', ax=ax, color='skyblue')

ax.set_title('Average Scores by Genre')
ax.set_ylabel('Average Score')
ax.set_xlabel('Genre')
ax.grid(axis='y', linestyle='--', linewidth=1.0)
plt.xticks(rotation=90, ha='center')
plt.tight_layout()

# %%
genre_stats = dataset.groupby('genre').agg({'score': 'mean', 'title': 'count'}).reset_index()
genre_stats = genre_stats.rename(columns={'score': 'avg_score', 'title': 'count_games'})

filtered_genre_stats = genre_stats[genre_stats['count_games'] > 500]

fig, ax = plt.subplots(figsize=(14, 8))

ax.scatter(filtered_genre_stats['count_games'], filtered_genre_stats['avg_score'], s=filtered_genre_stats['count_games'], alpha=0.5, edgecolors='black', linewidth=0.5)

for i, txt in enumerate(filtered_genre_stats['genre']):
    ax.annotate(txt, (filtered_genre_stats['count_games'].iloc[i], filtered_genre_stats['avg_score'].iloc[i]), fontsize=12, ha='center', va='center')

ax.set_title("Average Scores by Genre vs. Number of Reviews (Genres with Over 500 Reviews)")
ax.set_xlabel("Number of Games Reviewed")
ax.set_ylabel("Average Score")
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# %%
df = dataset.dropna()

platform_encoder = LabelEncoder()
df['platform'] = platform_encoder.fit_transform(df['platform'])

highest_score_data = df.groupby('release_year')['score'].transform(max)
df['is_top_game'] = (df['score'] == highest_score_data).astype(int)

train_data = df[df['release_year'] <= 2016]

x = train_data[['release_year', 'platform', 'score']]
y = train_data['genre']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

predicted_proba = clf.predict_proba([[2017, 1, 10]])
genre_probs = predicted_proba[0]

genres = clf.classes_

non_zero_indices = genre_probs > 0
filtered_genres = genres[non_zero_indices]
filtered_probs = genre_probs[non_zero_indices]

plt.figure(figsize=(12, 6))
plt.bar(filtered_genres, filtered_probs)
plt.xlabel('Genre')
plt.ylabel('Probability')
plt.title('Probability of Each Genre Being the Highest-Rated in 2017 (Excluding Zero Probabilities)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %%
