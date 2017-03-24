from __future__ import print_function
from pyspark import SparkContext, SparkConf
from operator import add
from pyspark.mllib.feature import HashingTF, IDF # TF-IDF specific functions 
import sys
import os

sc = SparkContext("local", "TF-IDF: Data analysis with Spark")
sc.setLogLevel("ERROR")
rddNeighbourID = sc.textFile('/usr/local/spark/spark-2.1.0-bin-hadoop2.7/listings_ids_with_neighborhoods', use_unicode = False).map(lambda x: x.split(","))

# full path to the folder with the datasets
folderPath = None

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




'''    ------------------ Functions under here  ---------------------	 '''

'''
Test built-in TF-IDF from pyspark
'''

mappedListing = listingsFiltered.map(lambda x: (x[getIndexValue("id")], x[getIndexValue("description")]))
joinedRDD = rddNeighbourID.join(mappedListing)
#print(joinedRDD.take(5))
reducedNeighbourhoodRDD = joinedRDD.map(lambda x: (x[1][0], x[1][1])).reduceByKey(add)
#print(reducedNeighbourhoodRDD.take(1))
reducedListingRDD = joinedRDD.map(lambda x: (x[0], x[1][1]))
#print(reducedListingRDD.take(1))

def heyListen(id):
    idRDD = reducedListingRDD.filter(lambda line: id in line).map(lambda x: x[1])
    #print(idRDD.collect())

#heyListen("1513847")

def heyNeighbor(neighborhood):
    neighborhoodRDD = reducedNeighbourhoodRDD.filter(lambda line: neighborhood in line).map(lambda x: x[1])
    print(neighborhoodRDD.collect())

#heyNeighbor("West Queen Anne")



# Checks if path to folder provided exists
def checkfolderPath(fn):
    fn = sys.argv[1]
    if os.path.exists(fn):
        # folderPath exists
        return True
    else:
        print(fn+ " does not exist. Retry with an existing path")
        return False

# check if folderpath ends on /
def formatFolderPath(fn):
    if fn[1].endswith('/'):
        return fn
    else:
        print(fn[1]+'/')
        return (fn[1]+'/')

# Checking flags, 
#   listing (-l) or a neighborhood (-n) 
#   should be analysed and listing id or neighborhood name on the input.
def flagPassing(args):
    for arg in range(len(args)):
        if args[arg]=='-l':
            listingID = args[arg+1]
            print('Flag -l accepted. Checking listingID: '+args[arg+1])




'''    ------------------ When running, under here  ---------------------	 '''

# Skeleton code for standalone application
'''
    # need to build all dependencies and modules
    # accepts a full path to the folder with the datasets,
    # a flag marking whether a listing (-l) or a neighborhood (-n) should be analysed and listing id or neighborhood name on the input.
'''
print("TF-IDF Assignment")
file = sc.textFile("/home/tin/Documents/BIGData/SkeletonCodeFromJAN/application_scaffolding/python_project/data.txt").cache()
print("File has " + str(file.count()) + " lines.")
print("Passed arguments " + str(sys.argv))
if (checkfolderPath(sys.argv)):
    folderPath=formatFolderPath(sys.argv)
    flagPassing(sys.argv)
    rddNeighbourID = sc.textFile(folderPath+'/6a_linkCoordinatesToNeighbourhood.csv',  use_unicode = False).map(lambda x: x.split(","))




# Don't know what this does yet. Taken from skeletonCode
sc.stop()
