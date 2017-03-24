from __future__ import print_function
from pyspark import SparkContext, SparkConf
from operator import add
from pyspark.mllib.feature import HashingTF, IDF # TF-IDF specific functions 

conf = SparkConf().setAppName("Data analysis with Spark")
sc = SparkContext(conf=conf)

rddNeighbourID = sc.textFile('/usr/local/spark/spark-2.1.0-bin-hadoop2.7/ANSWERS/6a_linkCoordinatesToNeighbourhood.csv', use_unicode = False).map(lambda x: x.split(","))

fsListings = sc.textFile('/usr/local/spark/spark-2.1.0-bin-hadoop2.7/listings_us.csv', use_unicode = False)
listingHeader = fsListings.first()
listingsFiltered = fsListings.filter(lambda x: x!=listingHeader).map(lambda x: x.split("\t"))

header2 = fsListings.first().split("\t")
dict = {}
for i in range(len(header2)):
    dict[header2[i]] = i

# getIndexValue then uses the key (column name) to access right column index.
def getIndexValue(name):
    return dict[name]

'''
Test built-in TF-IDF from pyspark
'''

mappedListing = listingsFiltered.map(lambda x: (x[getIndexValue("id")], x[getIndexValue("description")]))
joinedRDD = rddNeighbourID.join(mappedListing)
#print(joinedRDD.take(5))
reducedRDD = joinedRDD.map(lambda x: (x[1][0], x[1][1])).reduceByKey(add)
print(reducedRDD.take(1))