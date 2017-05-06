
import svm
from svmutil import *
import re, pickle, csv, os
import classifier_helper




def replaceTwoOrMore(s):
    # look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
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

def getSVMFeatureVector (tweet,featureList):
    sortedFeatures= sorted(featureList)
    test_feature_vector=[]
    for t in tweet:
        map1={}
        for w in sortedFeatures:
            map1[w] = 0

        tweet_word = t[0]

        for word in tweet_word:
            word = replaceTwoOrMore(word)
            word = word.strip('\'"?,.')
            if word in map1:
                map1[word] = 1

        values = list(map1.values())
        test_feature_vector.append(values)
        return test_feature_vector
def getSVMFeatureVectorAndLabels(tweets,featureList):
    sortedFeatures = sorted(featureList)
   # map1 = {}
    feature_vector = []
    labels=[]
    for t in tweets:
        label = 0
        map1 = {}
        #In1itialize empty map
        for w in sortedFeatures:
            map1[w] = 0

        tweet_words = t[0]
        tweet_opinion = t[1]
        #Fill the map
        for word in tweet_words:
            #process the word (remove repetitions and punctuations)
            word = replaceTwoOrMore(word)
            word = word.strip('\'"?,.')
            #set map[word] to 1 if word exists
            if word in map1:
                map1[word] = 1
                #values=1
                #feature_vector.append(values)
        #end for loop
        values =list (map1.values())
        #print(feature_vector)
        feature_vector.append(values)

        if tweet_opinion == 'positive':
            label = 0
        elif tweet_opinion == 'negative':
            label = 1
        elif tweet_opinion == 'neutral':
            label = 2
        labels.append(label)
    #    print(type(labels))

        #print(type(feature_vector))
    #return the list of feature_vector and labels
    return {'feature_vector': feature_vector, 'labels': labels}
#end
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

#classifierDumpFile = 'data/test/svm_test_model.pickle'
#Train the classifier
result = getSVMFeatureVectorAndLabels(tweets, featureList)

problem = svm_problem(result['labels'],result['feature_vector'])
#'-q' option suppress console output

param = svm_parameter('-q')
param.kernel_type = LINEAR
classifier = svm_train(svm_problem(result['labels'],result['feature_vector'], isKernel=True), param)
svm_save_model('model_file', classifier)


with open('data/b.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)


for i in your_list:
    if(i!=[]):
        test_feature_vector = getSVMFeatureVector(i, featureList)
        print(len(test_feature_vector))
        #print(your_list[2])
#p_labels contains the final labeling result
        p_labels, p_accs, p_vals = svm_predict([0] * len(test_feature_vector),test_feature_vector, classifier)
        print(p_labels)
        print(p_accs)