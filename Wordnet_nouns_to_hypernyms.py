#run using python2.7 hypertopics_new.py directoryname/
# transcribed_inputDirectoryName must exist
# this translates nouns into their most common hypernym class of either "physical_entity" or "abstraction"
#this can be used to test the ratio of abstract nouns versus physical entities or objectivity

import glob
import argparse
from nltk.corpus import wordnet as wn
from itertools import combinations
from multiprocessing import Process
import sys
import re
import argparse
import glob
reload(sys)
sys.setdefaultencoding("ISO-8859-1")

entity = wn.synset('entity.n.01') # The last entity in the tree

def print_hypernyms(synset,word_set):
    hypernyms = synset.hypernyms()
    #print hypernyms
    for hypernym in hypernyms:
        if hypernym != entity:
            word_set.add(hypernym.lemma_names()[0]) # note we only take the first synonym
            print_hypernyms(hypernym,word_set) # recursive call

parser = argparse.ArgumentParser()
parser.add_argument('file_dir') # the only argument is the directory of text files
args = parser.parse_args()
files = glob.glob(args.file_dir+'*')

for file in files:
    print 'Processing: '+file
    out_file = open('transcribed_'+file,'w') # transcribed_inputDirectoryName must exist
    for line in open(file,'r'):
        words = line.strip().split()
        line_result = ''
        for word in words:
            synsets = wn.synsets(word, pos=wn.NOUN) # get all noun synsets
            for word in synsets:
                word_set = set()
                #print word_set
                print_hypernyms(word,word_set) # find hypernyms for every definition
                if len(word_set)>0:
                    if len(line_result)>0:
                        line_result += ' '
                    line_result += ' '.join(word_set)
                if "physical_entity" in word_set:
                    #print "physical_entity"
                    line_result = "physical_entity"
                    break
                if "abstraction" in word_set:
                    #print "abstraction"
                    line_result = "abstraction"
                    break
                else:
                    line_result = ""
                print line_result
            if len(line_result)>0:
                out_file.write(line_result+'\n')
    out_file.close()
