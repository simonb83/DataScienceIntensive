from bs4 import BeautifulSoup
from urllib.request import urlopen
import csv
import re
import string


letters = list(string.ascii_uppercase)
movie_list = []

for l in letters:

    # Visit url
    url = "http://www.boxofficemojo.com/movies/alphabetical.htm?letter=" + l + "&p=.htm"
    soup = BeautifulSoup(urlopen(url).read(), "lxml")

    # Get sub-pages
    navbar = soup.find('div', 'alpha-nav-holder')
    pages = navbar.findAll('a', href=re.compile('alphabetical'))

    # Start with 1st page
    movietable = soup.find('div', {'id': 'main'})
    movies = movietable.findAll('a', href=re.compile('id'))
    for m in movies:
        movie_list.append([m["href"], re.search('(id=.+)', m["href"]).groups()[0], m.find('b').contents[0]])

    # Do subsequent pages for same letter if they exist
    if pages is not None:
        for p in pages:
            url = "http://www.boxofficemojo.com" + p.get("href")
            soup = BeautifulSoup(urlopen(url).read(), "lxml")
            # Scrape movies from current page
            movietable = soup.find('div', {'id': 'main'})
            movies = movietable.findAll('a', href=re.compile('id'))
            for m in movies:
                movie_list.append([m["href"], re.search('(id=.+)', m["href"]).groups()[0], m.find('b').contents[0]])

with open("movie_list.csv", "w") as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(['movieLink', 'movieID', 'name'])
    for row in movie_list:
        csvwriter.writerow(row)
