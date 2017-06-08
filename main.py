##############################
#          IMPORTS           #
##############################
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
import re
import numpy as np
import scipy as sp
import scipy.stats
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import date, timedelta
import urllib
from bisect import bisect_left
from xml.etree import ElementTree as ET
try:
    from matplotlib.finance import quotes_historical_yahoo_ochl
except ImportError:
    # quotes_historical_yahoo_ochl was named quotes_historical_yahoo before matplotlib 1.4
    from matplotlib.finance import quotes_historical_yahoo as quotes_historical_yahoo_ochl
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
from multiprocessing.pool import ThreadPool

##############################
#         CONSTANTS          #
##############################
stemmer = SnowballStemmer("english")
regex = re.compile('[^a-zA-Z]')
# Every database offered by ProQuest. Comment out ones you don't need to improve run time
databases = [
    'abidateline',
    'abiglobal',
    'abitrade',
    # 'artbibliographies',
    # 'artshumanities',
    # 'asfaaquaticpollution',
    # 'assia',
    # 'avery',
    'barrons',
    'blacknews',
    # 'cpi',
    # 'criminaljusticeperiodicals',
    # 'daai',
    # 'ebrary',
    # 'ecology',
    # 'education',
    # 'eisdigests',
    # 'envabstractsmodule',
    # 'environmentalengabstracts',
    # 'eric',
    'gannettnews',
    # 'healthmanagement',
    # 'healthsafetyabstracts',
    'hnpnewyorktimeswindex',
    # 'iba',
    # 'ibss',
    # 'libraryscience',
    # 'linguistics',
    # 'lisa',
    # 'llba',
    # 'microbiologya',
    # 'microbiologyb',
    # 'mlaib',
    'nationalnewspremier',
    # 'ncjrs',
    'nytimes',
    # 'pais',
    # 'pilots',
    # 'policyfile',
    # 'politicalscience',
    # 'pollution',
    # 'pqdtglobal',
    # 'pqrl',
    # 'psychology',
    # 'riskabstracts',
    # 'socabs',
    # 'socialservices',
    # 'sociology',
    # 'socscijournals',
    # 'ssamodule',
    # 'toxicologyabstracts',
    # 'toxline',
    # 'vogue',
    'wallstreetjournal',
    # 'waterresources',
    # 'wpsa'
]
oneDay = timedelta(days=1)


# Searches a list for a target. Return index of target if found. Otherwise, return -1
def binarySearch(list, target, lo=0, hi=None):  # can't use a to specify default for hi
    hi = hi if hi is not None else len(list)  # hi defaults to len(a)
    pos = bisect_left(list, target, lo, hi)  # find insertion position
    return (pos if pos != hi and list[pos] == target else -1)  # don't walk off the end


# Makes sure startDate and endDate don't fall on weekends (when stocks aren't traded)
def validateDates(startDate, endDate):
    # If you want to start on a weekend, start on the following Monday instead
    if (startDate.weekday() > 4):
        startDate += timedelta(days=(7 - startDate.weekday()))
    # If you want to end on a weekend, end on the last Friday instead
    if (endDate.weekday() > 4):
        endDate -= timedelta(days=(endDate.weekday() - 4))
    if (startDate >= endDate):
        raise ValueError('Invalid start and end dates. ' \
                         'Start date occurs after end date or maybe start date and end date occur on the same weekend.')
    return startDate, endDate

# There's no documentation or API for scraping proquest. Everything was trial and error.
# This part is very likely to break if ProQuest changes anything on their site.
def scrapeProquest(company, date):
    global databases
    data = []
    for database in databases:
        URL = 'http://fedsearch.proquest.com/search/sru/' + database + \
              '?operation=searchRetrieve&version=1.2&maximumRecords=10&startRecord=1&query=title%3D' + company + \
              '%20AND%20date%3D' + date.strftime("%Y%m%d")
        records = ET.parse(urllib.urlopen(URL)).getroot().find('{http://www.loc.gov/zing/srw/}records')
        for record in records.findall('{http://www.loc.gov/zing/srw/}record'):
            for datafield in record.find('{http://www.loc.gov/zing/srw/}recordData') \
                    .find('{http://www.loc.gov/MARC21/slim}record') \
                    .findall('{http://www.loc.gov/MARC21/slim}datafield'):
                if datafield.get('tag') == '245':
                    headline = datafield.find('{http://www.loc.gov/MARC21/slim}subfield').text
                    # Use regex to remove non-alphabetical characters
                    # Then use nltp's snowbkall stemmer to stem every word before storing it
                    stemmedHeadline = ' '.join([stemmer.stem(regex.sub('', word)) for word in headline.split(" ")])
                    data.append(stemmedHeadline + ' ')
                    # Alternative version, if you want to try skipping the regex & stemming
                    # data.append(headline + ' ')
    return data


# Get news headlines for every day from every source defined in databases[]
# You might need to be on RIT's network in order to authenticate yourself and be granted access to ProQuest
def getNews(company, newsDays):
    dailyHeadlines = defaultdict(list)
    threads = []
    pool = ThreadPool()
    for newsDay in newsDays:
        threads.append(pool.apply_async(scrapeProquest, (company, newsDay)))
    for i in range(len(newsDays)):
        dailyHeadlines[i] = threads[i].get()
        print 'Downloaded news for ' + company + ' on ' + newsDays[i].strftime("%Y%m%d")
    return dailyHeadlines


# Returns the mean and confidence interval for a list of data
def meanAndConfidenceInterval(data, confidence=0.95):
    array = 1.0 * np.array(data)
    size = len(array)
    mean = np.mean(array)
    standardError = scipy.stats.sem(array)
    deviation = standardError * sp.stats.t._ppf((1 + confidence) / 2., size - 1)
    return mean, mean - deviation, mean + deviation


def getVariations(companySymbol, startDate, endDate):
    # Get trading information for every company defined in companies{}
    quotes = [quotes_historical_yahoo_ochl(companySymbol, startDate - oneDay, endDate, asobject=True)]
    # Calculate the variance in trading prices
    variation = []
    tradingDays = []
    for quote in quotes:
        for i in range(0, quote.open.size):
            # Skip the first day because we can't compare it to anything
            # Also skip Friday->Monday because it's too large of a time gap
            if (i != 0) and (quote.date[i].weekday() != 0):
                # Append the difference between today's opening price and yesterday's
                variation.append(quote.open[i] - quote.open[i - 1])
                tradingDays.append(quote.date[i])
    return variation, tradingDays

# Acquire news to learn from or test with
def loadNews(companyName, tradingDays):
    fileName = companyName + '_' + tradingDays[0].strftime("%Y%m%d") + '-' + tradingDays[-1].strftime(
        "%Y%m%d") + '.json'
    dailyHeadlines = defaultdict(list)
    headlinesLoaded = False
    while (not headlinesLoaded):  # Loop until news successfully loaded
    # TODO Add timeout
        try:
            with open('news_' + fileName, 'r') as r:
                try:
                    dailyHeadlines = json.load(r, object_pairs_hook=OrderedDict)
                    print 'Cached news found and successfully loaded'
                    headlinesLoaded = True
                except ValueError:
                    print 'Error in loading cached news. Trying to re-download news.'
        except IOError:
            print 'news_' + fileName + ' not found. Downloading news.'
        if (not headlinesLoaded):
            dailyHeadlines = getNews(companyName, tradingDays)
            with open('news_' + fileName, 'w') as w:
                json.dump(dailyHeadlines, w, indent=2)

    # Combine every headline from that day into 1 big headline
    for day, text in dailyHeadlines.iteritems():
        dailyHeadlines[day] = "".join(text)
    return dailyHeadlines

def learn(companyName, companySymbol, startDate, endDate):
    # Confirm that start and end dates are good
    startDate, endDate = validateDates(startDate, endDate)

    variation, tradingDays = getVariations(companySymbol, startDate, endDate)

    # Acquire news to learn from
    dailyHeadlines = loadNews(companyName, tradingDays)

    # Use sklearn to calculate the tf-idf
    corpus = []
    for id, headlineToday in sorted(dailyHeadlines.iteritems(), key=lambda t: int(t[0])):
        corpus.append(headlineToday)
    tf = TfidfVectorizer(analyzer='word', stop_words='english')
    tfidf_matrix = tf.fit_transform(corpus)
    feature_names = tf.get_feature_names()
    dense = tfidf_matrix.todense()
    # Cross reference every word's tf-idf weighting with the amount of stock fluctuation that happened that day
    weightedWords = {}
    for i in range(0, len(dailyHeadlines)):
        dailyWeights = dense[i].tolist()[0]
        dailyMaxWeight = sum(dailyWeights)
        headline = dailyHeadlines[str(i)].split(' ')
        for word in headline:
            word_id = binarySearch(feature_names, word)
            if word_id != -1:
                if word in weightedWords:
                    weightedWords[word].append(variation[i] * dailyWeights[word_id])# / dailyMaxWeight)
                else:
                    weightedWords[word] = [variation[i] * dailyWeights[word_id]]# / dailyMaxWeight]
    # Calculate every word's average stock impact as well as the 95% confidence interval
    # Delete any word that only occurred once in the entire dataset because there isn't much significance to it
    for word, weights in weightedWords.items():
        if len(weights) > 1:
            weightedWords[word] = (meanAndConfidenceInterval(weights))
        else:
            del weightedWords[word]
    # Write final results to file
    fileName = companyName + '_' + tradingDays[0].strftime("%Y%m%d") + '-' + tradingDays[-1].strftime(
        "%Y%m%d") + '.json'
    with open('output_' + fileName, 'w') as f:
        print 'Saving final output to: ' + 'output_' + fileName
        json.dump(weightedWords, f, indent=2)
    return weightedWords

# Test using a dictionary and a single headline
# Outputs a prediction for the next day's fluctuation
def test(weightedWords, headline):
    avg = 0.0
    min = 0.0
    max = 0.0
    for word in headline.split(' '):
        stemmedWord = stemmer.stem(regex.sub('', word))
        # stemmedWord = word
        if stemmedWord in weightedWords:
            avg += weightedWords[stemmedWord][0]
            min += weightedWords[stemmedWord][1]
            max += weightedWords[stemmedWord][2]
    return avg, min, max

# Test using a dictionary, a company, and a range of dates
# Outputs a plot of the predicted variances over the range of dates
def test(weightedWords, companyName, companySymbol, startDate, endDate):
    # Confirm that start and end dates are good
    startDate, endDate = validateDates(startDate, endDate)

    variation, tradingDays = getVariations(companySymbol, startDate, endDate)

    # Acquire news to test with
    dailyHeadlines = loadNews(companyName, tradingDays)

    predictedAvgs = []
    predictedMins = []
    predictedMaxs = []
    for i in range(0, len(dailyHeadlines)):
        headline = dailyHeadlines[str(i)].split(' ')
        todayAvg = 0
        todayMin = 0
        todayMax = 0
        for word in headline:
            stemmedWord = stemmer.stem(regex.sub('', word))
            # stemmedWord = word
            if stemmedWord in weightedWords:
                todayAvg += weightedWords[stemmedWord][0]
                todayMin += weightedWords[stemmedWord][1]
                todayMax += weightedWords[stemmedWord][2]
        predictedAvgs.append(todayAvg)
        predictedMins.append(todayMin)
        predictedMaxs.append(todayMax)

    # Plot my predictions
    plottedYVals = tradingDays
    fig, ax = plt.subplots()
    ax.plot(tradingDays, predictedAvgs, label='Predicted', color='blue')
    # ax.plot(tradingDays, predictedMins, color='cyan')
    # ax.plot(tradingDays, predictedMaxs, color='cyan')
    ax.fill_between(tradingDays, predictedMins, predictedMaxs,
                    label='95% Confidence Interval', facecolor='cyan', alpha=0.3)
    ax.axhline(y=0, color='k')
    ax.plot(tradingDays, variation, label='Actual', color='red')
    # ax.fill_between()
    ax.legend(loc='lower right')
    plt.title(companyName + ' Stock Variations ('
              + startDate.strftime("%m/%d/%y") + '-' + endDate.strftime("%m/%d/%y") + ')')
    plt.show()

    # Calculate the percentage of correctly predicted price increases/decreases
    numCorrect = 0
    numTotal = 0
    for i in range(0, len(variation)):
        numTotal+=1
        if (variation[i] >= 0 and predictedAvgs[i] >= 0):
            numCorrect += 1
    print 'I got {}% of variations right'.format(float(numCorrect) * 100 / numTotal)

# Companies to use, identified by their ticker symbol
companies = {
    'GOOGL': 'Google'
}
# Duration of learning
learningStartDate = date(2000, 1, 1)
learningEndDate = date(2016, 1, 1)
# Duration of testing
testingStartDate = date(2016, 1, 1)
testingEndDate = date(2017, 1, 1)
for company in companies:
    dictionary = learn(companies[company], company, learningStartDate, learningEndDate)
    test(dictionary, companies[company], company, testingStartDate, testingEndDate)

