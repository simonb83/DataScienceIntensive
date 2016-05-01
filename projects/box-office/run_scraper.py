import time
import csv
from get_movie_data import movie_data

movies = []

with open("movie_list.csv", "r") as f:
    data = csv.reader(f)
    next(f)
    for d in data:
        movies.append(d)

general = []
details = []

# Chunk movie list into 1000 films at a time
chunks = [movies[x: x + 1000] for x in range(0, len(movies), 1000)]

for ch in chunks:
    for c in ch:
        gen, det = movie_data(c[1])
        general.append(gen)
        for d in det:
            details.append(d)
        # Pause 1 second between movies
        time.sleep(1)
    # Pause 5 mins between chunks
    time.sleep(5 * 60)

with open('general-info.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerow(['movieID', 'AdjustedDomesticGross', 'ReleaseDate', 'Distributor', 'Genre', 'Budget'])
    for row in general:
        writer.writerow(row)

with open('weekend-info.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerow(['movieID', 'Date', 'Rank', 'WeekendGross', 'pcChange','Theaters','Change','Avg','Gross-to-Date','Rank'])
    for row in details:
        writer.writerow(row)
