# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:35:49 2017

@author: thiag
"""

import xml.etree.cElementTree as ET
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime

from pylab import rcParams
rcParams['figure.figsize'] = 8, 6
rcParams['figure.dpi'] = 200
rcParams['font.size'] = 22

def getfileslist(folder):
    import os
    import zipfile
    import xml.etree.ElementTree as ET

    fileslist = os.listdir(folder)
    goodlist = badlist = []
    goodlist = [x for x in fileslist if x.endswith('.zip')]
    badlist = [x for x in fileslist if not x.endswith('.zip')]

#test each xml for parsing capabilities
    for filename in goodlist:
        try:
            archive = zipfile.ZipFile((folder + filename), 'r')
            if (archive.namelist()[0][-3:]=='xml')| \
                (archive.namelist()[0][-3:]=='XML'):
                cvfile = archive.open(archive.namelist()[0], 'r')
                ET.parse(cvfile)
            else:
                print('Error: file ' + archive.namelist()[0] + \
                      'is not a xml file.')
        except:
            print('XML parsing error in file ' + filename)
            goodlist.remove(filename)
            badlist.append(filename)
    return [goodlist, badlist]

folder="D:\\thiag\\Documents\\INPE\\Research\\Datasets\\" + \
        "DoutoresEngenharias\\Eng2\\"

[goodlist, badlist] = getfileslist(folder)

del(badlist)

summarylist = []

#def getdesc(filename):
for cvzipfile in goodlist:
    filename = folder + cvzipfile
#abre o arquivo zip baixado do site do lattes
    archive = zipfile.ZipFile((filename), 'r')
#cvdata = archive.read('curriculo.xml')
    cvfile = archive.open('curriculo.xml', 'r')

#get the summary information from lattes cv
    tree = ET.parse(cvfile)
    root = tree.getroot()
    try:
        desc = root[0][0].attrib['TEXTO-RESUMO-CV-RH']
    except:
        desc = ""
    summarylist.append(desc)
    
#initialize the tf idf matrix

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1),
                     min_df = 0, stop_words = ['quot'])

#fit and transform the list of lattes cv summaries to tf idf matrix

tfidf_matrix =  tf.fit_transform(summarylist)
feature_names = tf.get_feature_names() 

len(feature_names)

dense = tfidf_matrix.todense()

lattessummary = dense[0].tolist()[0]

#if the score of the word is >0, add the word and its score to the wordscores
#list

wordscores = [pair for pair in zip(range(0, len(lattessummary)), lattessummary)
              if pair[1] > 0]

#sort the score list by the score (second term)

sorted_wordscores = sorted(wordscores, key=lambda t: t[1] * -1)

topwords = []

for word, score in [(feature_names[word_id], score) for (word_id, score) 
                    in sorted_wordscores][:50]:
    print('{0: <50} {1}'.format(word, score))
    topwords.append(word)
    
x = ' '.join(topwords)
wordcloud = WordCloud().generate(x)
plt.axis('off')
plt.imshow(wordcloud, interpolation='bilinear')
plt.savefig(folder + 'wordcloud'+ datetime.now().strftime('%Y%m%d%H%M%S') + 
            '.png')
plt.show()