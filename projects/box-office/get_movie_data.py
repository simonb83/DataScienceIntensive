from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import Request
from random import choice
import re


HEADERS = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36"},
    {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'},
    {"User-Agent": 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'},
    {"User-Agent": 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12'}
]

def movie_data(filmID):
    url = "http://www.boxofficemojo.com/movies/?page=weekend&" + filmID + "&adjust_yr=2015&p=.htm"
    headers = choice(HEADERS)
    page = Request(url, headers=headers)

    soup = BeautifulSoup(urlopen(page).read(), "lxml", from_encoding="utf-8")
    body = soup.find('div', {'id': 'body'})

    general = [filmID]
    detailed = []

    # Domestic Total Gross
    domestic_total = body.find(string=re.compile("Domestic Total"))
    if domestic_total and domestic_total.find_parent():
        dtg = body.find(string=re.compile("Domestic Total")).find_parent().b.string
    else:
        dtg = "NA"
    general.append(dtg)
    # Release Date
    release_date = body.find(string=re.compile("Release Date"))
    if release_date and release_date.find_parent() and release_date.find_parent().a:
        rel_date = body.find(string=re.compile("Release Date")).find_parent().a["href"]
        if re.search('&date=(.+)&', rel_date):
            dte = re.search('&date=(.+)&', rel_date).groups()[0]
        else:
            dte = "NA"
    else:
        dte = "NA"
    general.append(dte)
    # Distributor
    distrib = body.find(string=re.compile("Distributor"))
    if distrib and distrib.find_parent():
        distributor = body.find(string=re.compile("Distributor")).find_parent().a.string
    else:
        distributor = "NA"
    general.append(distributor)
    # Genre
    gen = body.find(string=re.compile("Genre"))
    if gen and gen.find_parent():
        genre = body.find(string=re.compile("Genre")).find_parent().b.string
    else:
        genre = "NA"
    general.append(genre)
    # Production Budget
    production_budget = body.find(string=re.compile("Production Budget"))
    if production_budget and production_budget.find_parent():
        budget = body.find(string=re.compile("Production Budget")).find_parent().b.string
    else:
        budget = "NA"
    general.append(budget)

    # Weekend table data
    main = body.find('table', {'class': 'chart-wide'})
    if main:
        if main.find_parent() and main.find_parent().b and main.find_parent().b.font:
            year = main.find_parent().b.font.string
        else:
            year = "NA"

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
