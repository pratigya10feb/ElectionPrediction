# import regex
import re
import csv
import pprint
import nltk.classify


# start replaceTwoOrMore
def replaceTwoOrMore(s):
    # look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


# end

# start process_tweet
def processTweet(tweet):
    # process the tweets

    # Convert to lower case
    tweet = tweet.lower()
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    # Convert @username to AT_USER
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # trim
    tweet = tweet.strip('\'"')
    return tweet


# end

# start getStopWordList
def getStopWordList(stopWordListFileName):
    # read the stopwords
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords


# end

# start getfeatureVector
def getFeatureVector(tweet, stopWords):
    featureVector = []
    words = tweet.split()
    for w in words:
        # replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        # strip punctuation
        w = w.strip('\'"?,.')
        # check if it consists of only words
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        # ignore if it is a stopWord
        if (w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector


# end

# start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features


# end


# Read the tweets one by one and process it
inpTweets = csv.reader(open('data/215.csv', 'r'), delimiter=',')
stopWords = getStopWordList('data/feature_list/stopwords.txt')
count = 0
featureList = []
tweets = []
for row in inpTweets:
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment))
# end loop

# Remove featureList duplicates
featureList = list(set(featureList))

# Generate the training set
training_set = nltk.classify.util.apply_features(extract_features, tweets)

# Train the Naive Bayes classifier
#NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
MaxEntClassifier = nltk.classify.maxent.MaxentClassifier.train(training_set, max_iter = 10)#, 'GIS', trace=3, encoding=None, labels=None, sparse=True, gaussian_prior_sigma=0, max_iter = 10)
#testTweet = 'just had some bloodwork done. My arm hurts'

list_of_parties=['BJP','Congress','SP','BSP']
iTweet= csv.reader(open('data/b.csv', 'r'),delimiter=',')
#myfile = open('data/maxent_result.csv', 'w')
#wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
fp = open('final_maxent.txt', 'w')
#testTweet = 'Disappointing day. Attended a car boot sale to raise some funds for the sanctuary, made a total of 88p after the entry fee - sigh '
for row in iTweet:
   # print(row
#for row in iTweet:

    #testTweet = (String)row
    input_list = []
    testTweet = ''.join(str(e) for e in row)
    #processedTweet = processTweet(tweet)
    #featureVector = getFeatureVector(processedTweet, stopWords)
   # featureList.extend(featureVector)
   # tweets.append((featureVector, sentiment));

    if(testTweet!=' ,'):
        processedTestTweet = processTweet(testTweet)
        sentiment = MaxEntClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
        if(list_of_parties[0] in testTweet):
            print("testTweet = %s, sentiment = %s\n" % (testTweet, sentiment))
            writeStr = list_of_parties[0] + " | " + sentiment + "\n"
            fp.write(writeStr)
        elif(list_of_parties[1] in testTweet):
            print("testTweet = %s, sentiment = %s\n" % (testTweet, sentiment))
            writeStr = list_of_parties[1] + " | " + sentiment + "\n"
            fp.write(writeStr)
        elif (list_of_parties[2] in testTweet):
            print("testTweet = %s, sentiment = %s\n" % (testTweet, sentiment))
            writeStr = list_of_parties[2] + " | " + sentiment + "\n"
            fp.write(writeStr)
        elif (list_of_parties[3] in testTweet):
            print("testTweet = %s, sentiment = %s\n" % (testTweet, sentiment))
            writeStr = list_of_parties[3] + " | " + sentiment + "\n"
            fp.write(writeStr)


#processedTestTweet = processTweet(testTweet)
#print (MaxEntClassifier.classify(extract_features(getFeatureVector(processedTestTweet,stopWords))))

# Output
# =======
# positive

#print informative features
print( MaxEntClassifier.show_most_informative_features(10))

