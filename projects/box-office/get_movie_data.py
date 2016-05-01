from bs4 import BeautifulSoup
from urllib.request import urlopen
import re


def movie_data(filmID):
    url = "http://www.boxofficemojo.com/movies/?page=weekend&" + filmID + "&adjust_yr=2015&p=.htm"

    soup = BeautifulSoup(urlopen(url).read(), "lxml")
    body = soup.find('div', {'id': 'body'})

    general = [filmID]
    detailed = []

    # Domestic Total Gross
    if body.find(string=re.compile("Domestic Total")):
        dtg = body.find(string=re.compile("Domestic Total")).find_parent().find('b').contents[0]
    else:
        dtg = "NA"
    general.append(dtg)
    # Release Date
    if body.find(string=re.compile("Release Date")):
        rel_date = body.find(string=re.compile("Release Date")).find_parent().find('a')["href"]
        general.append(re.search('&date=(.+)&', rel_date).groups()[0])
    else:
        general.append("NA")
    # Distributor
    if body.find(string=re.compile("Distributor")):
        distributor = body.find(string=re.compile("Distributor")).find_parent().find('a').contents[0]
    else:
        distributor = "NA"
    general.append(distributor)
    # Genre
    if body.find(string=re.compile("Genre")):
        genre = body.find(string=re.compile("Genre")).find_parent().find('b').contents[0]
    else:
        genre = "NA"
    general.append(genre)
    # Production Budget
    if body.find(string=re.compile("Production Budget")):
        budget = body.find(string=re.compile("Production Budget")).find_parent().find('b').contents[0]
    else:
        budget = "NA"
    general.append(budget)

    # Weekend table data
    main = body.find('table', {'class': 'chart-wide'})
    if main:
        year = main.find_parent().find('b').find('font').contents[0]

        rows1 = main.findAll('tr', {'bgcolor': '#ffffff'})
        rows2 = main.findAll('tr', {'bgcolor': '#f4f4ff'})

        for r in rows1:
            data = [filmID, year]
            fields = r.findAll('td')
            data.append(str(fields[0].b.string).replace("\x96", "-"))
            for f in fields[1:]:
                t = f.find('font')
                if t.find('font'):
                    data.append(t.find('font').contents[0])
                else:
                    data.append(t.contents[0])
            detailed.append(data)
        for r in rows2:
            data = [filmID, year]
            fields = r.findAll('td')
            data.append(str(fields[0].b.string).replace("\x96", "-"))
            for f in fields[1:]:
                t = f.find('font')
                if t.find('font'):
                    data.append(t.find('font').contents[0])
                else:
                    data.append(t.contents[0])
            detailed.append(data)

    return general, detailed
