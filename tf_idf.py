from __future__ import print_function
from pyspark import SparkContext, SparkConf
from operator import add
from pyspark.mllib.feature import HashingTF, IDF # TF-IDF specific functions 
import sys
import os

sc = SparkContext("local", "TF-IDF: Data analysis with Spark")
sc.setLogLevel("ERROR")
rddNeighbourID = sc.textFile('/usr/local/spark/spark-2.1.0-bin-hadoop2.7/listings_ids_with_neighborhoods.tsv', use_unicode = False).map(lambda x: x.split("\t"))

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




mappedListing = listingsFiltered.map(lambda x: (x[getIndexValue("id")], x[getIndexValue("description")]))
joinedRDD = rddNeighbourID.join(mappedListing)
#print(joinedRDD.take(5))
reducedNeighbourhoodRDD = joinedRDD.map(lambda x: (x[1][0], x[1][1])).reduceByKey(add)
#print(reducedNeighbourhoodRDD.take(1))
reducedListingRDD = joinedRDD.map(lambda x: (x[0], x[1][1]))
#print(reducedListingRDD.take(1))




'''    ------------------ Functions under here  ---------------------	 '''
# filter for listing id
def heyListen(id):
    idRDD = reducedListingRDD.filter(lambda line: id in line).map(lambda x: (x[0], x[1].\
                                    replace(",","").\
                                    replace("(","").\
                                    replace(")","").\
                                    lower()))
    return idRDD


#heyListen("1513847")

# filter for neighborhood
def heyNeighbor(neighborhood):
    neighborhoodRDD = reducedNeighbourhoodRDD.filter(lambda line: neighborhood in line).map(lambda x: (x[0], x[1].\
                                                    replace(",", "").\
                                                    replace("(", "").\
                                                    replace(")", "").\
                                                    lower()))
    return neighborhoodRDD

#heyNeighbor("West Queen Anne")

def descriptionInTable(table):
    descriptionDict = {}
    for i in range(0, table.count()):
        descriptionDict[table.collect()[i][0]] = table.\
        flatMap(lambda x: x[1].strip().split()).\
        collect()
    print(descriptionDict)
    return descriptionDict

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
            listingAndDescription = descriptionInTable(heyListen(listingID))
            tfIDF(listingAndDescription[listingID])
        elif args[arg] == '-n':
            neighbor = args[arg+1]
            print('Flag -n accepted. Checking neighborhood: '+neighbor)
            descriptionInTable(heyNeighbor(neighbor))

######################### Task 1.1.1 TF-IDF
'''
Test built-in TF-IDF from pyspark
'''
def tfIDF(rdd):
    rdd = sc.parallelize(rdd)
    #rdd = rdd.map(lambda (listingID, text)
    # Read description words as TF vectors
    tf = HashingTF()
    tfVectors = tf.transform(rdd).cache()
    print(tfVectors)
    # Compute the IDF, then the TF-IDF vectors
    idf = IDF()
    idfModel = idf.fit(tfVectors)
    tfIdfVectors = idfModel.transform(tfVectors)
    print(tfIdfVectors.collect())
    


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
    rddNeighbourID = sc.textFile(folderPath+'listings_ids_with_neighborhoods.tsv',  use_unicode = False).map(lambda x: x.split("\t"))




# Don't know what this does yet. Taken from skeletonCode
sc.stop()
