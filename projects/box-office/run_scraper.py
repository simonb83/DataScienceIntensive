import time
import csv
from get_movie_data import movie_data
import urllib


def write_files(general, details):
    with open('general-info.csv', "a", newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(['movieID', 'AdjustedDomesticGross', 'ReleaseDate', 'Distributor', 'Genre', 'Budget'])
        for row in general:
            writer.writerow(row)

    with open('weekend-info.csv', "a", newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(['movieID', 'Year', 'Date', 'Rank', 'WeekendGross', 'pcChange','Theaters','Change','Avg','Gross-to-Date','Week_num'])
        for row in details:
            writer.writerow(row)


movies = []

with open("movie_list.csv", "r") as f:
    data = csv.reader(f)
    next(f)
    for d in data:
        movies.append(d)

# Skip the first N movies
n = 1199
new_movies = movies[n + 1:]


# Chunk movie list into 200 films at a time
chunks = [new_movies[x: x + 200] for x in range(0, len(movies), 200)]
# indices = [j for j in range(len(movies))]
# chunks = [[movies[i] for i in np.random.choice(indices, 100)]]

# Set which chunk we need to process
# start = 0

for ch in chunks:
    for c in ch:
        general = []
        details = []
        try:
            gen, det = movie_data(c[1])
            general.append(gen)
            for d in det:
                details.append(d)
            write_files(general, details)
        except urllib.error.HTTPError:
            break
        else:
            # Pause 3 seconds between movies
            time.sleep(3)
    time.sleep(30)
