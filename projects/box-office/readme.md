# Movie Data Visualization

##### Background:

During a discussion with my wife about the recent Batman vs. Superman film, the question came up as to whether the box office drop from the opening to the 2nd weekend was normal, or whether it dropped more than it should have due to bad word of mouth.

I decided to get hold of some data from the BoxOffice Mojo website to try and answer this question, and since then it has also evolved into an exercise in data visualization.

##### The Data:

I built two web-scrapers in order to obtain a large enough dataset to explore:
- The first was simply to crawl through the alphabetical film list on BoxOfficeMojo and create a list of films along with their url
- The second then went through the list film-by-film and scraped what data was available for:
    - Release Date
    - Adjusted Domestic Gross Takings (adjusted to 2015 prices)
    - Budget information
    - Genre
    - Weekend box office data such as Gross Takings, Rank, # of Cinemas etc.

Initially I had some issues with the second scraper due to missing pages, and also being booted from the site, but after making some adjustments to have a randomized wait time between urls, and also rotating between different user agents, I obtained a fairly complete dataset.

In total, of an initial list of 16,078 films, I was able to get general information for 16,065 of them along with weekend box office data for 13,023 films.

I had to do a fair amount of data cleaning in order to standardize some of the columns, extract budget information from text, and parse Release Dates in different formats to extract the Release Year and Release Month.

One key issue was that whereas I had collected Box Office Takings data in 2015 prices, none of the budget data was adjusted, and so I had to do further work to convert budgets to 2015 USD using inflation rates by year.

The final, cleaned dataset that I used is contained in all_cleaned_data.csv.

##### Known Issues With the Data:

The data is by no means complete or 100% clean. For instance:

- There is a lack of information for older films, and indeed the list is probably missing a large number of films pre-1990
- For many films, data exists for some fields, but not for others, and so for instance a film may appear very successful based on Box Office takings, but won't even show up in the list when film budget is considered
- For the key metrics I looked at, the data coverage is:
    - Total Domestic Gross: 88%
    - Adjusted Budget: 18%
    - Release Year: 89%
    - Opening w/end gross: 78%
    - Week 1 drop: 57%
    - Complete data: 14%

However, given that this started out as a fun, personal project, I decided that it was not worth investing additional time in complementing and cleaning the dataset further.

##### Key questions asked:

- What are some different measures of success?
- Which are the top 10 most successful films?
- What are the most successful genres of films?
- In which months are more successful films typically released?
- What has happened to film budgets and box office takings over time?
- Overall, how do films typically perform between their opening and second weekend?

Then, specifically for looking at Batman vs. Superman:

- What is an appropriate subset of films for comparison?
- How did Batman vs. Superman perform in its opening weekend comparatively?
- How did Batman vs. Superman compare in terms of drop between 1st and 2nd weekends?
- How much more money could Batman vs. Superman have made had it performed similarly to comparative films or sets of films?

##### The Analysis:

All the analysis is available in the following Jupyter notebooks:

- Top Level Analysis: High_Level_Story.ipynb
- Detailed Analysis: Detailed_Analysis.ipynb

