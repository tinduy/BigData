from __future__ import print_function
from pyspark import SparkContext, SparkConf
from operator import add
from pyspark.mllib.feature import HashingTF, IDF # TF-IDF specific functions 
import sys
import os
import operator
from time import time

sc = SparkContext("local", "TF-IDF: Data analysis with Spark")
sc.setLogLevel("ERROR")
rddNeighbourID = sc.textFile('/usr/local/spark/spark-2.1.0-bin-hadoop2.7/listings_ids_with_neighborhoods.tsv', use_unicode = True).map(lambda x: x.split("\t"))

# full path to the folder with the datasets
folderPath = None

fsListings = sc.textFile('/usr/local/spark/spark-2.1.0-bin-hadoop2.7/listings_us.csv', use_unicode = True)
listingHeader = fsListings.first()
listingsFiltered = fsListings.filter(lambda x: x!=listingHeader).map(lambda x: x.split("\t"))

header2 = fsListings.first().split("\t")
dict = {}
for i in range(len(header2)):
    dict[header2[i]] = i

# getIndexValue then uses the key (column name) to access right column index.
def getIndexValue(name):
    return dict[name]


mappedListing = listingsFiltered.map(lambda x: (x[getIndexValue("id")], x[getIndexValue("description")]))
joinedRDD = rddNeighbourID.join(mappedListing)
reducedNeighbourhoodRDD = joinedRDD.map(lambda x: (x[1][0], x[1][1])).reduceByKey(add)
reducedListingRDD = joinedRDD.map(lambda x: (x[0], x[1][1]))

'''    ------------------ Functions under here  ---------------------	 '''
# filter for listing id
def heyListen(id):
    idRDD = reducedListingRDD.filter(lambda line: id in line).map(lambda x: (x[0], x[1].\
                                    replace(",","").\
                                    replace("(","").\
                                    replace(")","").\
                                    replace("*", "").\
                                    replace(".", " ").\
                                    replace("-", "").\
                                    replace("!", "").\
                                    replace("+", "").\
                                    replace("/", " "). \
                                    replace("\\", " ").\
                                    replace("=", "").\
                                    replace("{", "").\
                                    replace("}", ""). \
                                    encode('utf-8'). \
                                    lower()))
    return idRDD

# filter for neighborhood
def heyNeighbor(neighborhood):
    neighborhoodRDD = reducedNeighbourhoodRDD.filter(lambda line: neighborhood in line).map(lambda x: (x[0], x[1].\
                                                    replace(",", "").\
                                                    replace("(", "").\
                                                    replace(")", ""). \
                                                    replace("*", ""). \
                                                    replace(".", " "). \
                                                    replace("-", ""). \
                                                    replace("!", ""). \
                                                    replace("+", ""). \
                                                    replace("/", " "). \
                                                    replace("\\", " ").\
                                                    replace("=", "").\
                                                    replace("{", "").\
                                                    replace("}", ""). \
                                                    encode('utf-8').\
                                                    lower()))
    return neighborhoodRDD

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
        return fn[1]
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
            start_time = time()
            idf2(tf(heyListen(listingID)), 1)
            print("Checking listing Elapsed time: " + str(time() - start_time))
            #tfIDF(listingAndDescription[listingID])
        elif args[arg] == '-n':
            neighbor = args[arg+1]
            print('Flag -n accepted. Checking neighborhood: '+neighbor)
            start_time = time()
            idf2(tf(heyNeighbor(neighbor)), 2)
            print("Checking neighbourhood Elapsed time: " + str(time() - start_time))

######################### Task 1.1.1 TF-IDF
'''
Test built-in TF-IDF from pyspark
'''

def tf(table):
    hello = table.\
        flatMap(lambda x: x[1].strip().split()).\
        map(lambda x: (x, int(1))).\
        reduceByKey(add)
    numberOfTermsInDocument = hello.count()
    print(numberOfTermsInDocument)
    haha = hello.map(lambda x: (x[0], float(float(x[1])/float(numberOfTermsInDocument))))
    return haha

def idf2(words, which):
    if (which == 1):
        numberOfDocuments = reducedListingRDD.count()
        weNeedThis = mappedListing.map(lambda x: (x[0], x[1].\
                                    replace(",", "").\
                                    replace("(", "").\
                                    replace(")", "").\
                                    replace("*", "").\
                                    replace(".", " ").\
                                    replace("-", " ").\
                                    replace("!", "").\
                                    replace("+", " ").\
                                    replace("/", " ").\
                                    replace("\\", " ").\
                                    replace("'s", " ").\
                                    replace("=", " ").\
                                    replace("{", " ").\
                                    replace("}", " "). \
                                    encode('utf-8').\
                                    lower()))
    elif (which == 2):
        numberOfDocuments = reducedNeighbourhoodRDD.count()
        weNeedThis = reducedNeighbourhoodRDD.map(lambda x: (x[0], x[1].\
                                                            replace(",", " ").\
                                                            replace("(", " ").\
                                                            replace(")", " ").\
                                                            replace("*", " ").\
                                                            replace(".", " ").\
                                                            replace("-", " ").\
                                                            replace("!", " ").\
                                                            replace("+", " ").\
                                                            replace("/", " "). \
                                                            replace("\\", " "). \
                                                            replace("'s", " ").\
                                                            replace("=", " ").\
                                                            replace("{", " ").\
                                                            replace("}", " "). \
                                                            encode('utf-8').\
                                                            lower()))
    detteErBra = weNeedThis.map(lambda x: (x[0], x[1].strip().split())).\
                                    flatMapValues(lambda x: x).\
                                    distinct().\
                                    map(lambda x: (x[1], int(1))).\
                                    reduceByKey(add).\
                                    map(lambda x: (x[0], float(float(numberOfDocuments)/float(x[1]))))
    joinRDD = words.join(detteErBra).\
                map(lambda x: (x[0], x[1][0] * x[1][1])).\
                map(lambda x: (x[1], x[0])).\
                sortByKey(0,1).\
                map(lambda x: (x[1], x[0]))
    print(joinRDD.take(100))

#### NEED TO WRITE TO FILE



'''    ------------------ When running, under here  ---------------------	 '''

# Skeleton code for standalone application
'''
    # need to build all dependencies and modules
    # accepts a full path to the folder with the datasets,
    # a flag marking whether a listing (-l) or a neighborhood (-n) should be analysed and listing id or neighborhood name on the input.
'''
print("TF-IDF Assignment")
#file = sc.textFile("/home/tin/Documents/BIGData/SkeletonCodeFromJAN/application_scaffolding/python_project/data.txt").cache()
#print("File has " + str(file.count()) + " lines.")
print("Passed arguments " + str(sys.argv))
if (checkfolderPath(sys.argv)):
    folderPath=formatFolderPath(sys.argv)
    flagPassing(sys.argv)

# Don't know what this does yet. Taken from skeletonCode
sc.stop()
