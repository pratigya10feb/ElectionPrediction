import nltk
from nltk.classify import *


import classifier_helper
from classifier_helper import *


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


pos_tweets = [('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative')]

tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    tweets.append((words_filtered, sentiment))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
      features['contains(%s)' % word] = (word in document_words)
    return features
word_features = get_word_features(get_words_in_tweets(tweets))
#nltk.set_word_features(word_features)
training_set = nltk.classify.apply_features(extract_features, tweets)
test = 1

classifier = nltk.NaiveBayesClassifier.train(training_set)
tweet = 'Im happy'
print( classifier.classify(extract_features(tweet.split())))
print (nltk.classify.accuracy(classifier, training_set))
classifier.show_most_informative_features(10)

    
