import time
import csv
import urllib
from get_movie_data import movie_data
from random import choice


def write_files(general, details):
    """
    Writes film data to separate CSV files
    general: List of strings
    details: List of strings

    returns: None
    """

    with open('general-info.csv', "a", newline='') as f:
        writer = csv.writer(f)
        for row in general:
            writer.writerow(row)

    with open('weekend-info.csv', "a", newline='') as f:
        writer = csv.writer(f)
        for row in details:
            writer.writerow(row)


# Choices for breaks (seconds) between requesting subsequent urls.
between_url_breaks = [2, 2, 2, 3, 3, 6]

# Choices for breaks (seconds) between processing subsequent batches of films.
between_batch_breaks = [10, 20, 30, 40]

movies = []

# Name of file to use for film urls
file_name = "missing-date.csv"

with open(file_name, "r") as f:
    data = csv.reader(f)
    next(f)
    for d in data:
        movies.append(d)

# Skip the first N movies
n = -1
new_movies = movies[n + 1:]


# Chunk movie list into 200 films at a time
chunks = [new_movies[x: x + 200] for x in range(0, len(movies), 200)]

error_links = []

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
            error_links.append([c[1]])
            continue
        else:
            # Pause between each movie
            time.sleep(choice(between_url_breaks))
    time.sleep(choice(between_batch_breaks))

with open("links_with_errors.csv", "a", newline='') as f:
    writer = csv.writer(f)
    for e in error_links:
        writer.writerow(e)
