from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np

f = pd.read_csv('Artifacts_Table_NON_Top.csv')
f2 = f.to_numpy()


def get_hypernym_tree(synset):

    for s in synset:

    return synset


for row in f2:
    word = row[0]
    synset = wn.synsets(word)
    get_hypernym_tree(synset)


dog = wn.synsets('glass')[1]
p = dog.hypernyms()
