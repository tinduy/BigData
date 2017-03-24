from __future__ import print_function
from pyspark import SparkContext, SparkConf
from operator import add
from pyspark.mllib.feature import HashingTF, IDF # TF-IDF specific functions 

conf = SparkConf().setAppName("Data analysis with Spark")
sc = SparkContext(conf=conf)



rddListing = sc.textFile('/home/tin/Documents/BIGData/airbnb_datasets/listings_us.csv', use_unicode = False)

dict = {}
for i in range(len(header2)):
    dict[header2[i]] = i

# getIndexValue then uses the key (column name) to access right column index.
def getIndexValue(name):
    return dict[name]
'''
Test built-in TF-IDF from pyspark
'''


rddListing = rddListing. 
