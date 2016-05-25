import pandas as pd

scraped = pd.read_csv('general-info.csv').drop_duplicates()
complete = pd.read_csv('movie_list.csv').drop_duplicates()

all_movies = complete.merge(scraped, how='left', indicator=True)
missing = all_movies[all_movies['_merge'] == 'left_only']
missing = missing.drop_duplicates()[['movieLink', 'movieID', 'name']]
missing.to_csv('missing-movies.csv', index=False)

print(len(missing))
