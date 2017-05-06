import svm
from svmutil import *
import featur
import csv

#with open('data/b.csv', 'r') as f:
 # reader = csv.reader(f)
  #your_list = list(reader)

#print(your_list)
#print(type(your_list))
#inpTweets = csv.reader(open('data/sampleTweets.csv', 'r'), delimiter=',')
#your_list = (inpTweets)
   # reader = csv.reader(f)
   # your_list = list(reader)
#for i in your_list:
#your_list=["i am hurt","i am happy","i am sad"]
#print(type(your_list[0]))
trainingDataFile = 'data/215.csv'
#trainingDataFile = 'data/full_training_dataset.csv'
classifierDumpFile = 'data/test/svm_test.pickle'
trainingRequired = 1
with open('data/b.csv', 'r') as f:
    reader = csv.reader(f,delimiter=',')
    your_list = list(reader)
    #print(your_list)
my_list=[]
for i in range(len(your_list)):
    if(your_list[i]!=[]):
       my_list.append(your_list[i])
print(my_list)
#for i in your_list:
 #   print(i)
  #  if(i!=[]):

    #for i in your_list:
sc = featur.SVMClassifiernew(my_list, trainingDataFile, classifierDumpFile, trainingRequired)
sc.classify()
sc.accuracy()
sc.writeOutput('finall.txt')