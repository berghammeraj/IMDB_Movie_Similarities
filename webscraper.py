""" Create a webscraper to get the summary and the metascore"""

from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from time import sleep, time
from random import randint
from IPython.core.display import clear_output
from warnings import warn

#Make a list for the url reference
#starts at 251 then iterates by +250 (501, 751, etc.)

url_nums = ['1']

for i in range(1, 40):
    val = i*250+1
    url_nums.append(str(val))

# Build the scraper
titles = []
years = []
certificates = []
runtimes = []
genres = []
imdb_ratings = []
mscores = []
summaries = []
num_votes = []
grosses = []

start_time = time()
requests = 0

for num in url_nums:

    # Make the get request
    response = get('https://www.imdb.com/search/title?title_type=feature&release_date=1970-01-01,2019-05-01&user_rating=5.0,10.0&languages=en&count=250&start=' + num + '&ref_=adv_nxt')

    # Sleep to not overload the server
    sleep(randint(8, 15))

    # Monitor the requests
    requests += 1
    elapsed_time = time() - start_time
    print('Request:{}; Frequency: {} requests/s'.format(requests, requests / elapsed_time))
    clear_output(wait=True)

    # throw a warning for a non-200 status code
    if response.status_code != 200:
        warn('Request: {}; Status code: {}'.format(requests, response.status_code))

    # Break the loop if the number of requests is greater than expected
    if requests > len(url_nums):
        warn('Number of requests was greater than expected!')
        break

    # Parse content of request with Beautiful Soup
    page_html = BeautifulSoup(response.text, 'html.parser')

    # Select all 250 movie containers froma  single page
    mv_containers = page_html.find_all('div', class_='lister-item mode-advanced')

    # Loop through the movies in each page
    for container in mv_containers:
        if container.find('div', class_='ratings-metascore') is not None:

    # Titles
            title = container.h3.a.text
            titles.append(title)

   # Years
            year = container.h3.find('span', class_='lister-item-year text-muted unbold').text
            years.append(year)

   # Certificates - parental rating (pg-13, R, etc.)
            certificate = container.p.span.text
            certificates.append(certificate)

   # Runtime - in minutes
            runtime = container.p.find('span', class_='runtime').text
            runtimes.append(runtime)

   # Genres - up to 3
            genre = container.p.find('span', class_='genre').text
            genres.append(genre)

   # IMDB ratings
            imdb_rating = float(container.strong.text)
            imdb_ratings.append(imdb_rating)

   # Metacritic scores
            mscore = container.find('span', class_='metascore').text
            mscores.append(mscore)

   # Summaries
            summary = container.find_all('p', class_='text-muted')[1].text
            summaries.append(summary)

   # Number of votes
            num_vote = container.find_all('span', attrs={'name': 'nv'})[0]
            num_vote = num_vote['data-value']
            num_votes.append(num_vote)

   # Gross
            try:
                gross = container.find_all('span', attrs={'name': 'nv'})[1]
                gross = gross['data-value']
            except:
                gross = None
            grosses.append(gross)


#Make it a dataframe
df = pd.DataFrame({'title':titles,
                   'year':years,
                   'certificate':certificates,
                   'runtime':runtimes,
                   'genre':genres,
                   'imdb_rating':imdb_ratings,
                   'meta_score':mscores,
                   'summary':summaries,
                   'num_votes':num_votes,
                   'gross':grosses})

#Write to csv
df.to_csv("data/scraped_data.csv")
