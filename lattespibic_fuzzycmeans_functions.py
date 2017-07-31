# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 21:44:46 2017

@author: thiag
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from bs4 import BeautifulSoup

from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from pylab import rcParams
rcParams['figure.figsize'] = 8, 6
rcParams['figure.dpi'] = 200
rcParams['font.size'] = 22

def lattesage(rawdata):
    from datetime import datetime
    x = [datetime.today()-datetime.strptime(str(i),'%d%m%Y') \
         for i in rawdata['atualizado']]
    y = np.array([round(i.days/30) for i in x])

#histograma da idade dos curriculos lattes
    plt.figure(figsize=(8,6), dpi=200)
    dummie = plt.hist(y, bins=range(0,round(max(y)+10, 2)),
                      align='left', histtype='bar', rwidth=0.95)

    plt.axvline(y.mean(), color='black', linestyle='dashed',
                linewidth=2)
    plt.text(round(max(y)+10, 2)*0.3, 0.8*max(dummie[0]),
             'Mean = ' + str(round(y.mean(),2)))            
    plt.suptitle('Age of Lattes CV in months', fontsize=20)
    plt.show()
    del dummie


def lattespibics(rawdata):

#histograma de quantas vezes o aluno fez bolsa PIBIC
    plt.figure(figsize=(8,6), dpi=200)

    plt.hist(rawdata['quantasVezesPIBIC'], 
             bins=range(max(rawdata['quantasVezesPIBIC'])+2), align='left',
             histtype='bar', rwidth=0.95)

    plt.xticks(range(min(rawdata['quantasVezesPIBIC']),
                     max(rawdata['quantasVezesPIBIC'])+1), fontsize=22)



    plt.suptitle('Scientific Initiation Grants per Student', fontsize=20)
    plt.show()

def mastersrateyear(rawdata):
#Histograma dos mestrandos
    plt.figure(figsize=(8,6), dpi=200)

    #If the first year is zero, get the second smaller year
    if min(rawdata['anoPrimeiroM']):
        anomaster0 = min(rawdata['anoPrimeiroM'])
    else:
        anomaster0 = sorted(set(rawdata['anoPrimeiroM']))[1]
#last year of occurence of a masters degree
    anomaster1 = max(rawdata['anoPrimeiroM'])

#plot the histogram and store in x
    dummie = plt.hist(rawdata['anoPrimeiroM'], bins=range(anomaster0,
                      anomaster1), align='left', histtype='bar', rwidth=0.95)
 
    plt.suptitle('Masters Degrees Obtained per Year', fontsize=20)

#Plot the total of people who finished masters degree in the given position
    plt.text(anomaster0, 0.8*max(dummie[0]), 'Total = ' +
             str(len(rawdata['anoPrimeiroM'].loc[lambda s:s>0])), size='20')
    plt.show()

    del dummie


def lattesgradlevel(rawdata):

    pibics = len(rawdata.loc[rawdata.quantasVezesPIBIC >= 1])
    masters = len(rawdata.loc[rawdata.quantosM >= 1])
    phds = len(rawdata.loc[rawdata.quantosD >= 1])
    pphds = len(rawdata.loc[rawdata.quantosPD >= 1])

    graddata = pd.DataFrame([pibics, masters, phds, pphds],
                            index=['Scientific Initiation','Masters','PhD',
                                   'Postdoctorate'],
                            columns=['Quantity'])


    fig = graddata.plot(y='Quantity', kind='bar', legend=False)
    fig.set_xticklabels(graddata.index, rotation=45)
    plt.title('Academic Level of the Students')

def getpubyeardata(rawdata):
#normalizar o ano da primeira publicacao
#o primeiro ano de publicacao e 2004
#ArtComp2004 = coluna 11
#TrabCong2004 = coluna 24

#Concatena a quantidade de artigos e de trabalhos em congressos apresentados
#como indice de produtividade
    pubyeardata = pd.DataFrame(index=rawdata.index)
    for i in range(0, 13):
#        pubyeardata['pub' + str(2004 + i)] = rawdata['ArtComp' +  
#                    str(2004 + i)] + rawdata['TrabCong' + str(2004 + i)]
        
        
        pubyeardata['pub' + str(2004 + i)] = rawdata['papers' +  
                    str(2004 + i)] + rawdata['works' + str(2004 + i)]
                                         
        pubdata = pubyeardata.copy()
        strindex = ['year']*pubdata.shape[1]
        for i in range(1,pubdata.shape[1]):
            strindex[i] = strindex[i] + str(i+1)
        pubdata.columns = strindex
    return pubdata

def firstnonzero(frame, nrow):
#shifta o primeiro indice para o primeiro ano em que houve producao
    n = frame.shape[1]
    count = 0
    while ((frame.iloc[nrow, 0] == 0) & (count < n)):
        for j in range(0, n-1):
            frame.set_value(nrow, frame.columns[j], frame.iloc[nrow,j+1])
        frame.set_value(nrow, frame.columns[n-1], 0)
        count = count + 1
        

def firstnonzero2(frame, frameindex, nrow):
    #shifta o primeiro indice para o primeiro ano do PIBIC
    n = frame.shape[1]
    count = 0
    nshift = frameindex[nrow] - 2004
    if nshift > 0:
        for j in range(0, n-1-nshift):
            frame.set_value(nrow, frame.columns[j], frame.iloc[nrow,j+nshift])
            frame.set_value(nrow, frame.columns[n-1-j], 0)
        count = count + 1

def setfuzzycmeansclstr(imin, imax, cleandata):

    fpcs = []
    centers = []
    clusters = []
    for i in range(imin, imax):
        center, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(\
            cleandata.transpose(), i, 2, error=0.005, \
            maxiter=1000, init=None)
        cluster_membership = np.argmax(u, axis=0)
#plota o histograma de cada centroide
        plt.figure()
    
        clusterweight = plt.hist(cluster_membership, bins=range(i+1),
                                 align='left', histtype='bar', rwidth=0.95)
        plt.xticks(range(0,i))
        plt.title('Number of Points in Each Centroid of the ' + 
                  str(i)+' Centroid Model')   
        plt.show()

#agrupa funcao de desempenho
        fpcs.append(fpc)
#agrupa os centroides
        centers.append(center)
#agrupa o peso dos centroides
        clusters.append(cluster_membership)
    
        fig, ax = plt.subplots()
        plt.title('Model with ' + str(i) + ' Mean Publication Profiles')
        for j in range(0,i):
            ax.plot(center[j], label=str(clusterweight[0][j]))
        
        legend = ax.legend(loc='upper right', shadow=True)
        plt.show()
    
#    plt.figure()
#    plt.plot(center, label=cluster_membership)
#    plt.title(str(i)+ ' Centroides')

    plt.figure()
    plt.plot(range(imin,imax),fpcs,'-x')
    plt.title('Fuzzy C Means Performance related to the Centroid Quantity')
    plt.show()
    return centers, clusters, fpcs

#file = "PIBIClattesframe.csv"
#folder= "D:\\thiag\\Documents\\INPE\\Research\\Lattes\\"

file = "dataframe.csv"
folder= "D:\\thiag\\Documents\\INPE\\Research\\Datasets\\DoutoresEngenharias\\"

filename = folder + file

rawdata = pd.read_csv(filename, engine='python')

cleandata = rawdata

lattesage(rawdata)

lattespibics(rawdata)

mastersrateyear(rawdata)

lattesgradlevel(rawdata)

pubdata = getpubyeardata(rawdata)

for i in range(0, pubdata.shape[0]-1):  
    firstnonzero2(pubdata, rawdata['anoPrimeiroPIBIC'], i)    
    
cleandata = pubdata
imin = 2
imax = 10
fpcs = []
centers = []
clusters = []

print('Rodada com todos os dados')

centers, clusters, fpcs = setfuzzycmeansclstr(imin, imax, cleandata)

#novo dataframe que recebe apenas os estudantes qeu publicaram
print('Rodada com todos os dados de alunos que publicaram pelo menos uma vez')
cleandata2 = cleandata[cleandata.sum(axis=1)!=0]
fpcs2 = []
centers2 = []
clusters2 = []

centers2, clusters2, fpcs2 = setfuzzycmeansclstr(imin, imax, cleandata2)

#remocao do outlier 287
#print('Rodada com todos os dados menos o outlier')
#cleandata3 = cleandata.drop(287)
#fpcs3 = []
#centers3 = []
#clusters3 = []

#centers3, clusters3, fpcs3 = setfuzzycmeansclstr(imin, imax, cleandata3)