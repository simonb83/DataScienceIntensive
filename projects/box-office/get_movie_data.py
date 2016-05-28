from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import Request
from random import choice
import re

# Choices of user agents for randomizing
HEADERS = [
    {"User-Agent": 'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko'},
    {"User-Agent": 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:41.0) Gecko/20100101 Firefox/41.0'},
    {"User-Agent": 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36'},
    {"User-Agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36'}
]


def movie_data(filmID):
    """
    Visits weekend info page on BoxOfficeMojo for a particular film and
    scrapes some general and detailed data.
    filmID: BoxOfficeMojo-specific film url id param in the format ?id=xxx.htm
    return: two separate lists of data; one-line for general information,
        and possibly multiple lines for detailed information.
    """

    url = "http://www.boxofficemojo.com/movies/?page=weekend&" + filmID + "&adjust_yr=2015&p=.htm"
    headers = choice(HEADERS)
    page = Request(url, headers=headers)

    soup = BeautifulSoup(urlopen(page).read(), "lxml", from_encoding="utf-8")
    body = soup.find('div', {'id': 'body'})

    if len(body.findAll(text=re.compile('Invalid Movie ID Specified.'))) > 0:
        return [], []

    general = [filmID]
    detailed = []

    # Domestic Total Gross
    dom = re.compile("Domestic Total")
    domestic_total = body.find(string=dom)
    if domestic_total and domestic_total.find_parent():
        dtg = body.find(string=dom).find_parent().b.string
    else:
        dtg = "NA"
    general.append(dtg)
    # Release Date
    rel = re.compile("Release Date")
    release_date = body.find(string=rel)
    if release_date and release_date.find_parent() and release_date.find_parent().a:
        rel_date = body.find(string=rel).find_parent().a["href"]
        if re.search('&date=(.+)&', rel_date):
            dte = re.search('&date=(.+)&', rel_date).groups()[0]
        else:
            dte = "NA"
    elif release_date and release_date.find_parent() and release_date.find_parent().nobr:
        dte = release_date.find_parent().nobr.string
    else:
        dte = "NA"
    general.append(dte)
    # Distributor
    dis = re.compile("Distributor")
    distrib = body.find(string=dis)
    if distrib and distrib.find_parent() and distrib.find_parent().a:
        distributor = body.find(string=dis).find_parent().a.string
    else:
        distributor = "NA"
    general.append(distributor)
    # Genre
    gr = re.compile("Genre")
    gen = body.find(string=gr)
    if gen and gen.find_parent():
        genre = body.find(string=gr).find_parent().b.string
    else:
        genre = "NA"
    general.append(genre)
    # Production Budget
    bud = re.compile("Production Budget")
    production_budget = body.find(string=bud)
    if production_budget and production_budget.find_parent():
        budget = body.find(string=bud).find_parent().b.string
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
