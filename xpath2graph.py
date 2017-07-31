# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 20:53:39 2017

@author: thiag
"""

import xml.etree.ElementTree as ET
import zipfile
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from pylab import rcParams
rcParams['figure.figsize'] = 8, 6
rcParams['figure.dpi'] = 200
rcParams['font.size'] = 22

folder= "C:\\Users\\thiag\\Desktop\\"
cvzipfile = "0096913881679975.zip" 

filename = folder + cvzipfile

def getgraph(filename):

#abre o arquivo zip baixado do site do lattes
    archive = zipfile.ZipFile((filename), 'r')
#cvdata = archive.read('curriculo.xml')
    cvfile = archive.open('curriculo.xml', 'r')

#inicializa o xpath
    tree = ET.parse(cvfile)
    root = tree.getroot()

## get all the authors names cited on lattes cv
#    x = root.findall('.//*[@NOME-COMPLETO-DO-AUTOR]')
#    nameattb = 'NOME-COMPLETO-DO-AUTOR'
## get all the phd commitees found in the lattes cv
    x = root.findall('.//PARTICIPACAO-EM-BANCA-DE-DOUTORADO/*[@NOME-DO-CANDIDATO]')
    nameattb = 'NOME-DO-CANDIDATO'
## get all commitees found in the lattes cv
#   x = root.findall('.//PARTICIPACAO-EM-BANCA-TRABALHOS-CONCLUSAO//*[@NOME-DO-CANDIDATO]')
#    nameattb = 'NOME-DO-CANDIDATO'
    y = list(enumerate(x))

#initialize the graph
    cvgraph = nx.Graph()

#get the cv owner name
    cvowner = root[0].attrib['NOME-COMPLETO']
    
#add the node representing the cv owner
    cvgraph.add_node(cvowner)

#for each name found in the cv
    for elem in x:
        dummie = elem.attrib[nameattb]
        if dummie != cvowner:
            if not(dummie in cvgraph.nodes()):
                cvgraph.add_node(dummie)
                cvgraph.add_edge(cvowner, dummie, weight=1)
            else:
                cvgraph[cvowner][dummie]['weight'] += 1
        
    return cvgraph

def topncontributions(xgraph, n):
    topcontribs = []
#takes the weights from the graph edges
    netweights = list([x[2]['weight'] for x in xgraph.edges(data=True)])
#creates a list with the sorted weights
    weightlist = list(np.sort(netweights)[::-1])

    for i in range(0,n):
        dummie = [k for k in xgraph.edges(data=True) if
                  k[2]['weight'] == weightlist[i]]
        for z in dummie: topcontribs.append(z)
    for z in topcontribs:
        print(xgraph.node[z[0]]['name'] + ' and ' + xgraph.node[z[1]]['name'] +
              ' have worked together ' + str(z[2]['weight']) + ' times.')
    networklarge = [x for x in network.edges(data=True) if 
                    x[2]['weight'] > weightlist[n]]
    nx.draw(nx.Graph(networklarge), with_labels=True)
    return topcontribs


grapha = getgraph(folder + "0096913881679975.zip")
graphb = getgraph(folder + "3413978291577451.zip")

network = nx.compose(grapha,graphb)


#to avoid plotting big names, convert names to integers, with a dictionary 
network = nx.convert_node_labels_to_integers(network, label_attribute='name')

pos=nx.spring_layout(network, k=1/np.sqrt(len(network)))
nx.draw(network, pos, with_labels='True')

plt.savefig(folder + 'graph'+ datetime.now().strftime('%Y%m%d%H%M%S') + '.png')

textfile = open(folder + "Labels" + datetime.now().strftime('%Y%m%d%H%M%S') + ".txt",
                "w")

for x in network.nodes(data=True):
    textfile.write(str(x[0])+ '\t'+ x[1]['name'] +'\n')

textfile.close()
del(x)

#list of all weights
netweights = list([x[2]['weight'] for x in network.edges(data=True)])

abc = topncontributions(network, 10)

network2 = nx.Graph(x for x in network.edges(data=True) if x[2]['weight']>1)

        