from __future__ import print_function
from pyspark import SparkContext, SparkConf
from operator import add
from pyspark.mllib.feature import HashingTF, IDF # TF-IDF specific functions 


sc = SparkContext("local", "TF-IDF: Data analysis with Spark")
sc.setLogLevel("ERROR")
rddNeighbourID = sc.textFile('/usr/local/spark/spark-2.1.0-bin-hadoop2.7/ANSWERS/6a_linkCoordinatesToNeighbourhood.csv', use_unicode = False).map(lambda x: x.split(","))

# Skeleton code for standalone application
'''
    # need to build all dependencies and modules
    # accepts a full path to the folder
    # with the datasets, a flag marking whether a listing (-l) or a neighborhood
    # (-n) should be analysed and listing id or neighborhood name on the input.
'''
print("TF-IDF Assignment")
#file = sc.textFile("/home/tin/Documents/BIGData/SkeletonCodeFromJAN/application_scaffolding/python_project/data.txt").cache()
#print("File has " + str(file.count()) + " lines.")
#print("Passed arguments " + str(sys.argv))
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


rddListing = rddListing.


# Don't know what this does yet. Taken from skeletonCode
sc.stop()

mappedListing = listingsFiltered.map(lambda x: (x[getIndexValue("id")], x[getIndexValue("description")]))
joinedRDD = rddNeighbourID.join(mappedListing)
#print(joinedRDD.take(5))
reducedRDD = joinedRDD.map(lambda x: (x[1][0], x[1][1])).reduceByKey(add)
print(reducedRDD.take(1))