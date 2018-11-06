# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 18:11:54 2017

@author: thiag
"""
####The code that runs goes here

#from consistency import consistency

from pylab import rcParams
rcParams['figure.figsize'] = 8, 6
rcParams['figure.dpi'] = 96
rcParams['font.size'] = 22

####This variable defines the amount of years of publications and works
####collected per Lattes CV files

NWORKS = 20

def get_freq_pie_chart(row, mytitle=""):
    """ Pie chart generating function

    From the list of numeric values in arg row generates a pie chart returning
    the chart data.

    Args:
        row(array): list of numeric values that generate the chart.
        mytitle(str): title of the chart.

    """
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure()
    from pylab import rcParams
    rcParams['figure.figsize'] = 8, 6
    rcParams['figure.dpi'] = 96
    rcParams['font.size'] = 12

    fig = plt.figure()

    ax = fig.add_subplot(221)
    
    labels, quants = np.unique(row.values, return_counts=True)
    percents = 100*quants/sum(quants)
    
    xyz = []
    for i in range(0, len(labels)):
        if len(str(percents[i]).partition('.')[0]) == 2:
            dummy = str(percents[i])[0:5]
        else:
            dummy = str(percents[i])[0:4]
        xyz.append([labels[i], quants[i], dummy + '%'])

    ax.set_title(mytitle)
    ax.axis("equal")

    pie = ax.pie(row.value_counts(), labels=['']*len(labels), startangle=90)

    ax2 = fig.add_subplot(211)
    ax2.axis("off")
    ax2.legend(pie[0], xyz, loc="center right")

    return xyz, fig

def get_ctgrs_pie_chart(row, mytitle="", refdate2=''):
    """ Pie chart generating function

    From the list of dates values in arg row generates a pie chart returning
    the chart data based on a list of categories (number of days the lattes cv
    was updated since today) implemented in the function

    Args:
        row(array): list of dates the lattes cvs have been updated.
        mytitle(str): title of the chart.

    """
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    from datetime import timedelta

    labels = ['0-30 days',
              '30-60 days',
              '60-90 days',
              '90-120 days',
              '120-150 days',
              '150-180 days',
              '180+ days']

    labels = np.array(labels)

    quants = np.array(7*[0])

    #verify usability of variable refdate2
    try:
        datetime.strptime(refdate2, '%d%m%Y')
    except:
        if refdate2.lower() == 'today':
            refdate2 = datetime.now().strftime('%d%m%Y')
        else:
            refdate2 = datetime.now().strftime('%d%m%Y')
            print('Upper-limit date invalid. Using default date of today.')

    mask = [(datetime.strptime(refdate2, '%d%m%Y') - \
             datetime.strptime(i, '%d%m%Y')).days for i in row]
        
    if min(mask) < 0:
        print('Dates more recent than date limit found. The most recent ' +
              'date is ' + str(-min(mask)) + ' days earlier than the '+
              'selected date and will be used as reference.')
        newref = min(mask)
        refdate2 = (datetime.strptime(refdate2, '%d%m%Y') +
                    timedelta(days=-newref)).strftime('%d%m%Y')
        mask = [i - newref for i in mask]
        
    refdate2 = datetime.strptime(refdate2, '%d%m%Y').strftime('%d/%m/%Y')
    for i in mask:
        if i < 0:
            print('Error. Reference date later than dates on the dataframe.')
            break
        elif i <= 30:
            quants[0] += 1
        elif i <= 60:
            quants[1] += 1
        elif i <= 90:
            quants[2] += 1
        elif i <= 120:
            quants[3] += 1
        elif i <= 150:
            quants[4] += 1
        elif i <= 180:
            quants[5] += 1
        else:
            quants[6] += 1

    percents = 100*quants/sum(quants)

    from pylab import rcParams
    rcParams['figure.figsize'] = 8, 6
    rcParams['figure.dpi'] = 96
    rcParams['font.size'] = 12

    fig = plt.figure()

    ax = fig.add_subplot(221)

    xyz = []
    for i in range(0, len(labels)):
        if len(str(percents[i]).partition('.')[0]) == 2:
            dummy = str(percents[i])[0:5]
        else:
            dummy = str(percents[i])[0:4]
        xyz.append([labels[i], quants[i], dummy + '%'])
    
    if mytitle == "":
        ax.set_title('Reference Date = '+ refdate2)
    else:
        ax.set_title(mytitle + '\nReference Date = '+ refdate2)
    ax.axis("equal")

    pie = ax.pie(quants, labels=['']*len(labels), startangle=90)

    ax2 = fig.add_subplot(211)
    ax2.axis("off")
    ax2.legend(pie[0], xyz, loc="center right")

    return xyz, fig

def word_list_to_cloud(cfolder, topwords, wtitle='', saveit=False):
    """ Word cloud generating function

    From the list of words in arg topwords generates a word cloud and saves
    in the folder passed through the arg folder.

    Args:
        folder(str): string that contains the folder address.
        topwords(list): list of the words that will generate the word
        cloud.
        wtitle(str): title of the word cloud
        saveit(bool): if True, saves the wordcloud file
    """

    from wordcloud import WordCloud
    from datetime import datetime
    import matplotlib.pyplot as plt
    import os

    folder = os.path.normpath(cfolder)

    dummy = ' '.join(topwords)
    wordcloud = WordCloud().generate(dummy)
    if wtitle != '':
        plt.title(wtitle)
    plt.axis('off')
    plt.imshow(wordcloud, interpolation='bilinear')
    if saveit:
        plt.savefig(folder + 'wordcloud'+
                    datetime.now().strftime('%Y%m%d%H%M%S') + '.png')
    plt.show()

def summary_list_top_words(summarylist, nwords=50, badwords=[], terms=[],
                           nprint=False):

    """ Gets words with higher TF-IDF score

    From the summarylist list of documents, gets the quantity nwords of words
    that have the biggest score according to TF-IDF algorithm.

    Args:
        summarylist (list): list of documents with its contents.
        nwords (int): quantity of words returned by the function.
        badwords (list): list of stopwords.
        terms: thesaurus/ vocabulary of terms.

    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
#initialize the tf idf matrix
#    badwords = ['quot','of','the','de','do','da','em','no','na']
    dummy = ['quot', 'of', 'the', 'in', 'and', 'on', 'at', 'for', 'to', 'by',
             'an', 'with', 'from', 'com', 'de', 'em', 'um', 'uma', 'do', 'da',
             'dos', 'das', 'para', 'no', 'na', 'nos', 'nas', 'por', 'pelo',
             'pela']

    wordcount = sum([len(x.split()) for x in summarylist])

    maxwordsloop = nwords

    if wordcount < nwords:
#        print('Number of words higher than number available. Using max' + \
#              'number of words.')
        maxwordsloop = wordcount

    badwords = badwords + list(set(dummy) - set(badwords))
    del dummy

    if terms == []:
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1),
                             min_df=0, max_df=.75, stop_words=badwords,
                             sublinear_tf=True)
    else:
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1),
                             min_df=0, max_df=.75, stop_words=badwords,
                             vocabulary=terms,
                             sublinear_tf=True)

#fit and transform the list of lattes cv summaries to tf idf matrix

    try:
        tfidf_matrix = tf.fit_transform(summarylist)
        feature_names = tf.get_feature_names()

        dense = tfidf_matrix.todense()

        lattessummary = np.sum(dense, axis=0).tolist()[0]

    #if the score of the word is >0, add the word and its score to the wordscores
    #list

        wordscores = [pair for pair in zip(range(0, len(lattessummary)),
                                           lattessummary) if pair[1] > 0]

    #sort the score list by the score (second term)

        sorted_wordscores = sorted(wordscores, key=lambda t: t[1] * -1)

        topwords = []

        for word, score in [(feature_names[word_id], score) for (word_id, score)
                            in sorted_wordscores][:maxwordsloop]:
            if nprint:
                print('{0: <40} {1}'.format(word, score))
            topwords.append(word)

        while len(topwords) < nwords:
            topwords.append('')
    except:
        topwords = [] #*nwords

    return topwords

def get_node_connections(nodelist, network, depth=1):

    """ Gets quantity of connections of first or second grade.

    From the list of nodes in arg nodelist, finds the number of direct
    connections (if depth equals 1) or of second degree connections (if depth
    equals 2) in the node network in the arg network.

    Args:
        nodelist: list of nodes to evaluate number of connections.
        network: network where the nodes will be searched and the number of
        connections will be counted.
        depth: depth of the quantity of connections. Equals 1 for direct
        connections and 2 for connections and connections of connections.

    """
    import networkx as nx
    if (depth > 2) | (depth < 1):
        print("invalid depth supplied")
    connections = []
    dummy = network.edges()

    if depth == 1:
        for i in range(0, len(nodelist)):
            x = 0
            for y in dummy:
                if nodelist[i] in y:
                    x = x + 1
            connections.append(x)

    elif depth == 2:
        for i in range(0, len(nodelist)):
            x = 0
#list all the edges connecting to node intlist[i]
            dummy2 = network.edges(nodelist[i])
            for y in dummy2:
#verify connections of those connected to intlist[i]
                for z in dummy:
                    if y[1] in z:
                        x = x + 1
            connections.append(x)

    return connections

def nodes_class(graph, nodeslist):

    """ Classify nodes in the graph network with reference to nodeslist.

    From the list of nodes in arg nodelist, verify if these nods are part of
    the network in arg graph. If they are, and attribute "type" is set as
    "internal". Otherwise, the attribute "type" is set as "external"

    Args:
        nodelist: list of nodes to compare with the network and to receive the
        attribute "type".
        graph: network where the nodes in nodelist will be searched and
        be considered "internal" or "external".

    """
    import networkx as nx
#    nx.set_node_attributes(graph, "type", "external")
    nx.set_node_attributes(graph, "external", "type")
    for x in graph.nodes():
        if x in nodeslist:
            graph.node[x]["type"] = "internal"
    return graph


def lattes_owner(cfolder, filename):

    """ Returns the name of the owner of the Lattes CV found in arg filename
    Args:
        filename: complete path (folder + file) of the file where the Lattes
        CV is found. The file must be a zip file containing a XML file.
    """
    import zipfile
    import xml.etree.ElementTree as ET
    import os

    folder = os.path.normpath(cfolder)
    rightfile = os.path.join(folder, filename)

    #opens the Lattes CV file
    archive = zipfile.ZipFile(rightfile, 'r')
    cvfile = archive.open('curriculo.xml', 'r')

    #initializes xpath
    tree = ET.parse(cvfile)
    root = tree.getroot()

    #get the cv owner name
    cvowner = root[0].attrib['NOME-COMPLETO']
    return cvowner

def lattes_age(rawdata, refdate=""):
    """ Plots a histogram of the Lattes CV collection age. The age is
        calculated from the date refdate.

    Args:
        rawdata: dataframe containing the data for the histogram.
        refdate: the date of reference used to calculate the age in days.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime

    try:
        date0 = datetime.strptime(refdate, '%d%m%Y')
    except:
        print('Date format invalid. Using todays date as reference.')
        date0 = datetime.today()

    from datetime import datetime
    x = [date0 - datetime.strptime(str(i), '%d%m%Y') \
         for i in rawdata['atualizado']]
    y = np.array([round(i.days/30) for i in x])

    z = y[y > 0]
#histogram of Lattes CV age in months
    plt.figure(figsize=(8, 6), dpi=96)
    dummy = plt.hist(z, bins=range(0, round(max(z)+10, 2)),
                     align='left', histtype='bar', rwidth=0.95)

    plt.axvline(z.mean(), color='black', linestyle='dashed',
                linewidth=2)
    plt.text(round(max(z)+10, 2)*0.3, 0.8*max(dummy[0]),
             'Mean = ' + str(round(z.mean(), 2)) + ' months', fontsize=16)
    plt.xlabel("Lattes CV age in months")
    plt.ylabel("Quantity of Lattes CVs")
    plt.suptitle('Age of Lattes CV in months', fontsize=20)
    plt.show()
    del dummy


def lattes_pibics(rawdata):

    """ Plots a histogram of PIBIC scholarships per researcher in the dataframe.

    Args:
        rawdata: dataframe containing the data for the histogram.
    """
    import matplotlib.pyplot as plt
#histogram of PIBIC scholarships per researcher
    plt.figure(figsize=(8, 6), dpi=96)

    vpibic = rawdata['quantasVezesPIBIC'].tolist()

    plt.hist(vpibic, bins=range(max(vpibic)+2), align='left',
             histtype='bar', rwidth=0.95)

    plt.xticks(range(min(vpibic), max(vpibic)+1), fontsize=22)
    
    plt.xlabel("Quantity of PIBICs grants")
    plt.ylabel("Number of researchers")

    plt.suptitle('Scientific Initiation Grants per Researcher', fontsize=20)
    plt.show()

def degree_rate_year(rawdata, degreetype='G'):
#Histogram of masters degrees
    """ Plots a histogram of a given degree concluded per year.

    Args:
        rawdata: dataframe containing the data for the histogram.
        degreetype: 'G' for graduation (default option)
                    'M' for masters degree
                    'D' for doctorate / PhD
                    'PD' for post-doctorate.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6), dpi=96)

    if degreetype.lower() == 'pd':
        vtitle = rawdata['anoPrimeiroPosDoc'].tolist()
        dummy1 = 'Post-Doctorate'
    elif degreetype.lower() == 'd':
        vtitle = rawdata['anoPrimeiroD'].tolist()
        dummy1 = 'Doctorate'
    elif degreetype.lower() == 'm':
        vtitle = rawdata['anoPrimeiroM'].tolist()
        dummy1 = 'Masters'
    elif degreetype.lower() == 'g':
        vtitle = rawdata['anoPrimeiraGrad'].tolist()
        dummy1 = 'Graduation'
    else:
        print('Invalid parameter "degreetype". Using default parameter "G".')
        vtitle = rawdata['anoPrimeiraGrad'].tolist()
        dummy1 = 'Graduation'

#If the first year is zero, get the second smaller year
    if min(vtitle):
        degyear0 = min(vtitle)
    else:
        degyear0 = sorted(set(vtitle))[1]
#last year of occurence of that degree

    degyear1 = max(vtitle)

#plot the histogram and store in x
    dummy2 = plt.hist(vtitle, bins=range(degyear0, degyear1),
                      align='left', histtype='bar', rwidth=0.95)
    
    plt.xticks(range(degyear0, degyear1, 5))

    plt.xlabel("Calendar year")
    plt.ylabel("Number of degrees obtained")

    plt.suptitle(dummy1 + ' Degrees Obtained per Year', fontsize=20)

#Plot the total of people who finished masters degree in the given position
    plt.text(degyear0, 0.98*max(dummy2[0]), 'Total = ' +
             str(len([i for i in vtitle if i > 0])), size='20')

#    plt.text(degyear0, 0.8*max(dummy2[0]), 'Total = ' +
#             str(len([i for i in vtitle if i > 0])), size='20')
    plt.show()

    del dummy2

def lattes_grad_level(rawdata):
    """ Plots a histogram of the specialization level of the researchers.

    Args:
        rawdata: dataframe containing the data for the histogram.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from pylab import rcParams
    rcParams['figure.figsize'] = 8, 6
    rcParams['figure.dpi'] = 96

    pibics = len(rawdata.loc[rawdata.quantasVezesPIBIC >= 1])
    grads = len(rawdata.loc[rawdata.quantasGrad >= 1])
    masters = len(rawdata.loc[rawdata.quantosM >= 1])
    phds = len(rawdata.loc[rawdata.quantosD >= 1])
    pphds = len(rawdata.loc[rawdata.quantosPD >= 1])

    graddata = pd.DataFrame([pibics, grads, masters, phds, pphds],
                            index=['Scientific Initiation', 'Graduations',
                                   'Masters', 'PhD', 'Postdoctorate'],
                            columns=['Quantity'])


    plt.figure(figsize=(8, 6), dpi=96)
    fig = graddata.plot(y='Quantity', kind='bar', legend=False)
    fig.set_xticklabels(graddata.index, rotation=45)
    plt.ylabel('Quantity of researchers')
    plt.title('Academic Level of the Researchers')
    plt.show()

def get_pub_year_data(rawdata):
    """ Arranges the publication data from calendar year to a generic sequence
        of years, preparing this data to be used for other means.

    Args:
        rawdata: dataframe containing the data for the histogram.
    """
    from datetime import datetime

    import pandas as pd

    firstyear = datetime.now().year - NWORKS + 1

    #Normalizes the year of the first publication
    #The first year of the publication dataframe is the current calendar
    #year minus NWORKS

    #For means of production indexes, the quantity of papers and works
    #presented in congresses are summed to each other

    pubyeardata = pd.DataFrame(index=rawdata.index)
    for i in range(0, NWORKS):
        pubyeardata['pub' + str(firstyear + i)] = \
        rawdata['papers' + str(firstyear + i)] + \
        rawdata['works' + str(firstyear + i)]

        if i == NWORKS-1:
            pubdata = pubyeardata.copy()
            strindex = ['year']*pubdata.shape[1]
            for i in range(1, pubdata.shape[1]):
                strindex[i] = strindex[i] + str(i+1)

    pubdata.columns = strindex

    return pubdata

def first_nonzero(frame, frameindex, option=0):
    """Changes the production per year vector based on the variable "option".
    Args:
        frame: the dataframe where the production data is found.
        frameindex: contains the year of the first PIBIC scholarship for each
           researcher in arg frame. Is only used if arg option=2
        option=0: production is based on calendar year. leave frame unchanged.
        option=1: first production value is first non-zero value of the
           production vector. The last vector indexes are substituted by zeros.
        option=2: first production year is first year of PIBIC scholarship.
           Only make sense to use it in PIBIC-based dataframes.
    """
    import pandas as pd
    from datetime import datetime

    firstyear = datetime.now().year - NWORKS + 1

    nrows = frame.shape[0]
    ncols = frame.shape[1]
    count = 0

    if option == 1:
        #for each row
        for i in range(0, nrows):
            count = 0
            while (frame.iloc[i, 0] == 0) & (count < ncols):
                for j in range(0, ncols-1):
#                    frame.set_value(i, frame.columns[j], frame.iloc[i, j+1])
                    frame.iat[i, j] = frame.iloc[i, j+1]
#                frame.set_value(i, frame.columns[ncols-1], 0)
                frame.iat[i, ncols-1] = 0
                count = count + 1

    elif option == 2:

        for i in range(0, nrows):
    #    nshift = frameindex[nrow] - 2004
            nshift = frameindex.iloc[i] - firstyear
            if nshift > 0:
                for j in range(0, ncols-1-nshift):
#                    frame.set_value(i, frame.columns[j], frame.iloc[i, j+nshift])
                    frame.iat[i, j] = frame.iloc[i, j+nshift]
#                    frame.set_value(i, frame.columns[ncols-1-j], 0)
                    frame.iat[i, ncols-1-j] = 0
    else:
        if option != 0:
            print('Invalid option in function argument.\n Using'+ \
                  ' default option = 0.')

def set_fuzzycmeans_clstr(imin, imax, cleandata):
    """Applies the algorithm fuzzy c means to the dataframe in arg cleandata
        to try to find a quantity of mean publication profiles. This quantity
        varies from arg imin to arg imax.
    Args:
        cleandata: the dataframe where the production data is found.
        imin: minimum quantity of fuzzy c means clusters to be evaluated.
        imax: maximum quantity of fuzzy c means clusters to be evaluated.
    """
    import matplotlib.pyplot as plt
    import skfuzzy as fuzz
    import numpy as np
    from sklearn import metrics
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import scale


    fpcs = []
    centers = []
    clusters = []
    for i in range(imin, imax):
        center, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(\
            cleandata.transpose(), i, 2, error=0.001, \
            maxiter=10000, init=None)
        cluster_membership = np.argmax(u, axis=0)
#plota o histograma de cada centroide
        plt.figure()

        clusterweight = plt.hist(cluster_membership, bins=range(i+1),
                                 align='left', histtype='bar', rwidth=0.95)
        plt.xticks(range(0, i))
        plt.title('Number of Researchers Associated to Each Centroid of the ' +
                  str(i)+' Centroid Model')
        plt.xlabel('Centroid label')
        plt.ylabel('Researchers associated to centroid')
        plt.show()
        
        ndim = int(np.ceil(np.sqrt(i)))
        if ndim*(ndim-1) >= i:
            fig, axs = plt.subplots(ndim-1, ndim, sharex=True, sharey=True)
        else:
            fig, axs = plt.subplots(ndim, ndim, sharex=True, sharey=True)
        
        # add a big axes, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
#        plt.tick_params(labelcolor='none', top='off', bottom='off',
#                        left='off', right='off')
        plt.tick_params(labelcolor='none', top=False, bottom=False,
                        left=False, right=False)
        plt.grid(False)
        plt.xlabel('Year index')
        plt.ylabel('Number of publications')
        
        fig.suptitle('Publication profiles and their associated Centroid')
        
        for j1 in range(ndim):
            for j2 in range(ndim):
                k = ndim*j1 + j2
                if k < i:
                    try:
                        axs[j1,j2].plot(range(cleandata.shape[1]),\
                                        cleandata[cluster_membership==k].T)
                    except:
                        pass
                    axs[j1,j2].plot(center[k], 'k', linestyle='-.', linewidth=2)
                    axs[j1,j2].grid(linestyle=':')
        
#agrupa funcao de desempenho
        fpcs.append(fpc)
#agrupa os centroides
        centers.append(center)
#agrupa o peso dos centroides
        clusters.append(cluster_membership)

        fig, ax = plt.subplots()
        fig.suptitle('Model with ' + str(i) + ' Mean Publication Profiles')
        for j in range(0, i):
            ax.plot(center[j], label=str(clusterweight[0][j]))

        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                  shadow=True)

#        legend = ax.legend(loc='right', shadow=True)
        
        plt.xlabel('Year index')
        plt.ylabel('Number of publications')

        plt.show()

    plt.figure()
    plt.plot(range(imin, imax), fpcs, '-x')
    plt.xlabel('Centroid Quantity')
    plt.ylabel('FPC')
    plt.title('Fuzzy Partition Coefficient (FPC) as function of Centroid' +
              ' Quantity\n Best = 1, Worst = 0')
    plt.show()
    return centers, clusters, fpcs


def get_grad_years(x, gradtype):
    """From the arg x, that contains a XML piece of the Lattes CV where a
    graduation course data is found, extract the number of years spent on the
    course, as well as the year in which the course has started.
    Args:
        gradtype: type of graduation course.
    """
    nfirst = nquant = 0
#There may be cases where the year of conclusion or the year of beggining
#of the course have not been disclosed. The function begins trying to find
#these inputs, to avoid errors.
    if gradtype != "LIVRE-DOCENCIA":
        try:
            if x.attrib["ANO-DE-CONCLUSAO"] != "":
                flag_end = True
            else:
                flag_end = False
        except:
            flag_end = False
    
        try:
            if x.attrib["ANO-DE-INICIO"] != "":
                flag_start = True
            else:
                flag_start = False
        except:
            flag_start = False
    
        if (flag_end) | (flag_start):
#if any year is given
            nquant = nquant + 1
            if nquant == 1:
                if not flag_end:
                    inityear = get_year_from_str(x.attrib["ANO-DE-INICIO"])
                    x.attrib["ANO-DE-CONCLUSAO"] = \
                        inityear + xpectd_years(gradtype)
    
                nfirst = get_year_from_str(x.attrib["ANO-DE-CONCLUSAO"])
            else:
                if not flag_end:
                    inityear = get_year_from_str(x.attrib["ANO-DE-INICIO"])
                    x.attrib["ANO-DE-CONCLUSAO"] = \
                        inityear + xpectd_years(gradtype)
    
                dummy = get_year_from_str(x.attrib["ANO-DE-CONCLUSAO"])
                if nfirst > dummy:
                    nfirst = dummy
                    
    elif gradtype == "LIVRE-DOCENCIA":
        try:
            nfirst = get_year_from_str(x.attrib["ANO-DE-OBTENCAO-DO-TITULO"])
            nquant = 1
        except:
            pass
        
        
#if no year (start or end) is given, the function returns the standard values
#of zero.
    return [nfirst, nquant]

def get_year_from_str(conclusionyear):
    """
    """
    try:
        dummy = int(conclusionyear)
    except:
    #if there is a '/' somewhere
        if '/' in conclusionyear:
            dummy = int(conclusionyear.split('/')[1])
            if dummy < 100:
                if dummy < 20:
                    dummy = dummy + 2000
                else:
                    dummy = dummy + 1900
        else:
            dummy = 9999
    return dummy

def xpectd_years(gradtype):
    """From the arg gradtype, returns an average number of years the
    graduation course found in gradtype takes to be concluded.
    Args:
        gradtype: type of graduation course.
    """
    if gradtype == "GRADUACAO":
        return 4
    elif gradtype == "MESTRADO":
        return 2
    elif gradtype == "DOUTORADO":
        return 4
    elif gradtype == "POS-DOUTORADO":
        return 2
    elif gradtype == "LIVRE-DOCENCIA":
        return 0
    else:
        print("Invalid input for graduation type in function xpectdyears")
        return 0

def get_graph_from_file(filename, opt='all'):
    """From the file in filename returns a graph where the nodes are the
    researchers and the edges are the collaborations between these researchers.
    Args:
        filename: contains the name of the file where the graph is extracted.
        opt==all: lists all names found in the Lattes CV as nodes.
        opt==phdboards: lists all names found where the researcher has taken
        part of PhD boards.
        opt==allboards: lists all names found where the researcher has taken
        part of graduation boards.
    """
#From input file filename gets all partnerships occurred between the owner of
#the Lattes CV of the filename file and phd presentations listed on that CV.

    import zipfile
    import xml.etree.ElementTree as ET
    import networkx as nx

#opens zip file downloaded from lattes website
#abre o arquivo zip baixado do site do lattes

    archive = zipfile.ZipFile(filename, 'r')
    cvfile = archive.open('curriculo.xml', 'r')

#inicializa o arquivo xpath
#initializes xpath file

    tree = ET.parse(cvfile)
    root = tree.getroot()

    if opt.lower() == 'all':
## get all the authors names cited on lattes cv
#        x = root.findall('.//*[@NOME-COMPLETO-DO-AUTOR]')
        x = root.findall('.//PRODUCAO-BIBLIOGRAFICA//*[@NOME-COMPLETO-DO-AUTOR]')
        nameattb = 'NOME-COMPLETO-DO-AUTOR'
    elif opt.lower() == 'phdboards':
## get all the phd commitees found in the lattes cv
        x = root.findall('.//PARTICIPACAO-EM-BANCA-DE-DOUTORADO/*[@NOME-DO-CANDIDATO]')
        nameattb = 'NOME-DO-CANDIDATO'
    elif opt.lower() == 'allboards':
## get all commitees found in the lattes cv
        x = root.findall('.//PARTICIPACAO-EM-BANCA-TRABALHOS-CONCLUSAO//*[@NOME-DO-CANDIDATO]')
        nameattb = 'NOME-DO-CANDIDATO'

#initialize the graph
    cvgraph = nx.Graph()

#get the cv owner name
    cvowner = root[0].attrib['NOME-COMPLETO']

#add the node representing the cv owner
    cvgraph.add_node(cvowner)

#for each name found in the cv
    for elem in x:
        dummy = elem.attrib[nameattb]
        if dummy != cvowner:
            if not(dummy in cvgraph.nodes()):
                cvgraph.add_node(dummy)
                cvgraph.add_edge(cvowner, dummy, weight=1)
            else:
                cvgraph[cvowner][dummy]['weight'] += 1

    return cvgraph

def get_graph_from_folder(folderlist, saveit=False):
    """From the folders in folderlist returns a graph where the nodes are the
    researchers and the edges are the collaborations between these researchers.
    Args:
        folderlist: the list of folder where the Lattes CV files are found. 
           The Lattes CV files are downloaded as .zip files containing a 
           .xml file.
        saveit: boolean. If True, saves the files generated by the function.
    """

    import LattesLab as ll
    import networkx as nx
    import matplotlib.pyplot as plt
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import os

    filelist = errlist = []
    [filelist, errlist] = get_files_list(folderlist)
    del errlist

    folder = os.getcwd()

    vecgraph = []
    namelist = []

    for filename in filelist:
        dummygraph = ll.get_graph_from_file(filename)
        vecgraph.append(dummygraph)
        namelist.append(ll.lattes_owner(os.path.dirname(filename), filename))

    #join the graphs in the vector vecgraph in a single network

    network = ll.join_graphs(vecgraph)

    network = ll.nodes_class(network, namelist)

    #to avoid plotting big names, convert names to integers, with a dictionary
    network = nx.convert_node_labels_to_integers(network,
                                                 label_attribute='name')

    intlist = []
    extlist = []
    for x in network.nodes(data=True):
        if x[1]['type'] == 'internal':
            intlist.append(x[0])
        elif x[1]['type'] == 'external':
            extlist.append(x[0])

    #nx.draw_networkx_nodes(network,pos=nx.spring_layout(network),nodelist=intlist)

    #pos=nx.spring_layout(network, k=1/np.sqrt(len(network)))
    pos = nx.shell_layout(network, [intlist, extlist])
    nx.draw_networkx_nodes(network, pos, nodelist=extlist,
                           node_color='r', alpha=0.2, 
                           label='Other researchers in the network')
    nx.draw_networkx_nodes(network, pos, nodelist=intlist, node_color='b',
                           label='Lattes CV Owners')
    #nx.draw_networkx_labels(network,pos)
    nx.draw_networkx_edges(network, pos, width=1.0, alpha=0.2)
    plt.axis('off')

    plt.legend(bbox_to_anchor=(1.05, 1), prop={'size': 12})
    plt.text(1.2,-1.2,
             'Number of Lattes CV Owners= ' + str(len(intlist)) +
             '\nNumber of other researchers = ' + str(len(extlist)),
             wrap=True, fontsize=12,
             bbox=dict(facecolor='white', edgecolor='black'))
    plt.show()

    if saveit:
        figfile = 'graph' + datetime.now().strftime('%Y%m%d%H%M%S') + '.png'
    
        plt.savefig(os.path.join(folder, figfile), format='png')

        textfilename = "Labels" + datetime.now().strftime('%Y%m%d%H%M%S') + ".txt"
    
        textfile = open(os.path.join(folder, textfilename), "w")
    
        for x in network.nodes(data=True):
            dummy1 = str(x[0]).encode(encoding='ISO-8859-1', errors='strict'). \
                            decode(encoding='utf-8', errors='ignore')
            dummy2 = x[1]['name'].encode(encoding='ISO-8859-1', errors='strict'). \
                            decode(encoding='utf-8', errors='ignore')
            textfile.write(dummy1 + '\t'+ dummy2 + '\n')
    
        textfile.close()

    #list of all weights
#    netweights = list([x[2]['weight'] for x in network.edges(data=True)])

    top_n_contributions(network, 10)

#    network2 = nx.Graph(x for x in network.edges(data=True) \
#                        if x[2]['weight'] > 1)

    #measure how many first connections people in internal group have:
    firstconnections = get_node_connections(intlist, network, 1)

    #measure how many second connections people in internal group have:
    secondconnections = get_node_connections(intlist, network, 2)
    
    ratios = np.array(firstconnections)/np.array(secondconnections)
    ratios[np.isnan(ratios)] = 0

    connections = pd.DataFrame(
        {'name': namelist,
         'firstconnections': firstconnections,
         'secondconnections': secondconnections,
         'ratio': ratios
        })
    datafilename = "DataFrame" + datetime.now().strftime('%Y%m%d%H%M%S') + \
                   ".txt"

    connections.to_csv(os.path.join(folder, datafilename))

    return [network, connections]

def top_n_contributions(xgraph, n):
    """From the graph found in arg xgraph, takes the n largest (with score
    repetition) weights between the edges, which means in the biggest
    contributions between two researchers found in the xgraph nodes.
    Args:
        xgraph: graph where the weight of the edges is measured.
        n: number of highest weights returned by the function.
    """
#from a given graph xgraph, gets the n biggest weights and returns

    import networkx as nx
    import numpy as np
    topcontribs = []

#takes the weights from the graph edges

    netweights = list([x[2]['weight'] for x in xgraph.edges(data=True)])

#creates a list with the sorted weights

    weightlist = list(reversed(np.unique(netweights)))
    count = 0

    for i in range(0, n):
        if count < n:
            dummy = [k for k in xgraph.edges(data=True) if
                     k[2]['weight'] == weightlist[i]]
            for z in dummy:
                topcontribs.append(z)
            count = count + len(dummy)

    for z in topcontribs:
        print(xgraph.node[z[0]]['name'] + ' and ' + xgraph.node[z[1]]['name'] +
              ' have worked together ' + str(z[2]['weight']) + ' times.')
#    networklarge = [x for x in xgraph.edges(data=True) if
#                    x[2]['weight'] > weightlist[n]]
#    nx.draw(nx.Graph(networklarge), with_labels=True)
    return topcontribs


def get_files_list(folderlist):
    """Gets a list of files from the arg folder and check for errors.
    Args:
        folderlist: the list of folder where the Lattes CV files are found. 
           The Lattes CV files are downloaded as .zip files containing a 
           .xml file.
    """

    import os
    import zipfile
    import xml.etree.ElementTree as ET

    if not isinstance(folderlist, list):
        folderlist = [folderlist]

    goodlist = []
    badlist = []

    for cfolder in folderlist:

        folder = os.path.normpath(cfolder)

        fileslist = [os.path.join(folder, x) for x in os.listdir(folder)]
        good_dummy = [x for x in fileslist if x.endswith('.zip')]
        bad_dummy = [x for x in fileslist if not x.endswith('.zip')]
        goodlist += good_dummy
        badlist += bad_dummy

#test each xml for parsing capabilities
    for filename in goodlist:
        try:
#            rightname = os.path.join(folder, filename)
            archive = zipfile.ZipFile(filename, 'r')
            if archive.namelist()[0][-3:].lower() == 'xml':
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

def get_colab(filename, columns):
    """From file in arg filename, extract the production of the Lattes CV owner
        and the other researchers that were coauthors in this production.
    Args:
        filename: file containing the Lattes CV subject to analysis.
        columns: list of columns that compose the dataframe.
    """
    import zipfile
    import xml.etree.ElementTree as ET
    import pandas as pd

#abre o arquivo zip baixado do site do lattes
    archive = zipfile.ZipFile(filename, 'r')
#cvdata = archive.read('curriculo.xml')
    cvfile = archive.open('curriculo.xml', 'r')

#inicializa o xpath
    tree = ET.parse(cvfile)
    root = tree.getroot()

    #cv owner id
    cvid = root.attrib['NUMERO-IDENTIFICADOR']

    colabframe = pd.DataFrame(columns=columns)

    #list of all works in events

    colabtype = 'T'

    xworks = root.findall('.//TRABALHO-EM-EVENTOS')

    for x in xworks:
        authors = x.findall('AUTORES')
        for y in authors:
            try:
                dummyid = y.attrib['NRO-ID-CNPQ']
            except:
                dummyid = ''
            if (dummyid != '') & (dummyid != cvid):
                cvid2 = y.attrib['NRO-ID-CNPQ']
                year = x[0].attrib['ANO-DO-TRABALHO']
                title = x[0].attrib['TITULO-DO-TRABALHO']
                dummy = pd.DataFrame(data=[[colabtype, cvid, cvid2, year, title]],
                                     columns=columns)
                colabframe = colabframe.append(dummy)

    #list of all papers
    colabtype = 'A'

    xpapers = root.findall('.//ARTIGO-PUBLICADO')

    for x in xpapers:
        authors = x.findall('AUTORES')
        for y in authors:
            try:
                dummyid = y.attrib['NRO-ID-CNPQ']
            except:
                dummyid = ''
            if (dummyid != '') & (dummyid != cvid):
                cvid2 = y.attrib['NRO-ID-CNPQ']
                year = x[0].attrib['ANO-DO-ARTIGO']
                title = x[0].attrib['TITULO-DO-ARTIGO']
                dummy = pd.DataFrame(data=[[colabtype, cvid, cvid2, year, title]],
                                     columns=columns)
                colabframe = colabframe.append(dummy)

    return colabframe

def join_graphs(vecgraph):
    """Joins two or more graphs found in the arg vecgraph.
    Args:
        vecgraph: list of graphs to be concatenated.
    """
    import networkx as nx
    if len(vecgraph) < 2:
        return vecgraph[0]
    elif len(vecgraph) == 2:
        return nx.compose(vecgraph[0], vecgraph[1])
    else:
        dummy = nx.compose(vecgraph[0], vecgraph[1])
        for i in range(2, len(vecgraph)):
            dummy = nx.compose(dummy, vecgraph[i])
        return dummy

def get_lattes_desc_folders(folderlist):
    """Extracts the description section of each Lattes CV found in arg folder.
    Args:
        folder: the folder where the Lattes CV files are found. The Lattes CV
            files are downloaded as .zip files containing a .xml file.
        savefile: if True, the dataframe is stored in a .csv file for posterior
            use.
    TODO: implement capacity to save file. Maybe use pickle
    """
    import zipfile
    import xml.etree.cElementTree as ET
    import LattesLab as ll

    [goodlist, badlist] = ll.get_files_list(folderlist)

    summarylist = []

    del badlist
    for filename in goodlist:
#abre o arquivo zip baixado do site do lattes
        archive = zipfile.ZipFile(filename, 'r')
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

    return summarylist

def get_dataframe_from_folders(folderlist, savefile=True):
    """Extracts the Lattes CV dataframe to be used by other functions in
        this library.
    Args:
        folderlist: a list of strings. Each string contains the name of one
            folder where Lattes CV files are found. The Lattes CV
            files are downloaded as .zip files containing a .xml file.
        savefile: if True, the dataframe is stored in a .csv file for posterior
            use.
    """
    import pandas as pd
    import xml.etree.ElementTree as ET
    import zipfile
    from datetime import datetime
    import os

#initiate the dataframe
    columns = ['Nome',
               'lattesId',
               'nacionalidade',
               'atualizado',
               'quantasVezesPIBIC',
               'anoPrimeiroPIBIC',
               'quantasGrad',
               'anoPrimeiraGrad',
               'quantosM',
               'anoPrimeiroM',
               'quantosD',
               'anoPrimeiroD',
               'quantosPD',
               'anoPrimeiroPosDoc'] + \
               ["works" + str(datetime.now().year - NWORKS + i + 1) for i in range(0, NWORKS)] + \
               ["papers" + str(datetime.now().year - NWORKS + i + 1) for i in range(0, NWORKS)]

    lattesframe = pd.DataFrame(columns=columns)

#verify if the parameter folderlist is a list

    if not isinstance(folderlist, list):
        folderlist = [folderlist]

#filters the zip files

    ziplist = nonziplist = []
    [ziplist, nonziplist] = get_files_list(folderlist)

    count = 0
    for rightname in ziplist:
        count += 1
        archive = zipfile.ZipFile(rightname, 'r')
        cvfile = archive.open(archive.namelist()[0], 'r')

        tree = ET.parse(cvfile)
        root = tree.getroot()

    #list all XML file attributes
        elemtree = []

        for elem in tree.iter():
            elemtree.append(elem.tag)

        elemtree = list(set(elemtree))

    #Retrieve genaral data
        name = root[0].attrib["NOME-COMPLETO"]
        try:
            readid = str(root.attrib["NUMERO-IDENTIFICADOR"])
        except:
            readid = str(9999999999999999)
        lastupd = str(root.attrib["DATA-ATUALIZACAO"])
#        print(name)
        try:
            nation = root[0].attrib["SIGLA-PAIS-NACIONALIDADE"]
        except:
            nation = "Unspecified"

    #Retrieve academic background data

        [ano1grad, ngrad] = get_grad_count(root, "GRADUACAO")
        [ano1master, nmaster] = get_grad_count(root, "MESTRADO")
        [ano1phd, nphd] = get_grad_count(root, "DOUTORADO")
        [ano1postdoc, nposdoc] = get_grad_count(root, "POS-DOUTORADO")
        [ano1livredoc, nlivredoc] = get_grad_count(root, "LIVRE-DOCENCIA")

    #RESEARCHER PRODUCTION

        root[1].getchildren()

    #WORKS PUBLISHED
        x = root.findall('.//TRABALHOS-EM-EVENTOS')

        if not x:
            qtyworks = 0
        else:
            qtyworks = len(x[0].getchildren())

        nprod = []

        for i in range(0, qtyworks):
            nprod.append(x[0][i][0].attrib["ANO-DO-TRABALHO"])

        nprodyear = [0]*NWORKS

    #For a interval of Nwork years, count number of publications per year
    #from current year backwards

        for i in range(0, NWORKS):
            nprodyear[i] = nprod.count(str(datetime.now().year - NWORKS + i + 1))

    #amout of papers published
        x = root.findall('.//ARTIGOS-PUBLICADOS')

        if len(x) > 0:
            npapers = len(x[0].getchildren())
        else:
            npapers = 0

        allpapers = []

        for i in range(0, npapers):
            allpapers.append(x[0][i][0].attrib["ANO-DO-ARTIGO"])

        npaperyear = [0]*NWORKS

    #For a interval of Nwork years, count number of publications per year
    #from current year backwards

        for i in range(0, NWORKS):
            npaperyear[i] = allpapers.count(str(datetime.now().year - NWORKS + i + 1))

    #Retrieve Scientific Initiation Scolarships
        x = root.findall('.//*[@OUTRO-VINCULO-INFORMADO="Iniciação Cientifica"]') + \
            root.findall('.//*[@OUTRO-VINCULO-INFORMADO="Iniciação Científica"]') + \
            root.findall('.//*[@OUTRO-ENQUADRAMENTO-FUNCIONAL-INFORMADO=' +  \
                               '"Bolsista de Iniciação Cientifica"]') + \
            root.findall('.//*[@OUTRO-ENQUADRAMENTO-FUNCIONAL-INFORMADO=' +  \
                               '"Bolsista de Iniciação Científica"]') + \
            root.findall('.//*[@OUTRO-ENQUADRAMENTO-FUNCIONAL-INFORMADO=' +  \
                               '"Aluno de Iniciação Cientifica"]') + \
            root.findall('.//*[@OUTRO-ENQUADRAMENTO-FUNCIONAL-INFORMADO=' +  \
                               '"Aluno de Iniciação Científica"]')

        qtdeanos = 0
        if not x:
            ano1PIBIC = 0
        else:
            ano1PIBIC = int(x[0].attrib["ANO-INICIO"])

        for elem in x:
            if elem.attrib["ANO-FIM"] == "": # & \
    #            (elem.attrib["SITUACAO"]=="EM_ANDAMENTO"):
#using last year of the interval as the last year that the CV has been updated
                elem.attrib["ANO-FIM"] = str(
                        datetime.strptime(lastupd, '%d%m%Y').year)
            if elem.tag == "VINCULOS":
                if (elem.attrib["MES-INICIO"] != "") & \
                (elem.attrib["MES-FIM"] != ""):
                    qtdeanos += (float(elem.attrib["ANO-FIM"]) -
        				   			  float(elem.attrib["ANO-INICIO"]) +
    					   		     (float(elem.attrib["MES-FIM"]) -
    						   	      float(elem.attrib["MES-INICIO"]))/12)
                else:

    #Sometimes the project begins and ends in the same year.
    #Correcting the delta for these cases.

                    if elem.attrib["ANO-FIM"] == elem.attrib["ANO-INICIO"]:
                        qtdeanos += 1
                    else:
                        qtdeanos += (float(elem.attrib["ANO-FIM"]) -
                                     float(elem.attrib["ANO-INICIO"]))
            if ano1PIBIC > int(elem.attrib["ANO-INICIO"]):
                ano1PIBIC = int(elem.attrib["ANO-INICIO"])

        qtdePIBIC = round(qtdeanos)

        x = [name, readid, nation, lastupd, qtdePIBIC, ano1PIBIC,
             ngrad, ano1grad, nmaster, ano1master, nphd,
             ano1phd, nposdoc, ano1postdoc] + nprodyear + npaperyear

        dummy = pd.DataFrame(data=[x], columns=columns)

        lattesframe = lattesframe.append(dummy)

#        print(str(100*count/len(ziplist)) + '%%')

#reindex the dataframe
    lattesframe = lattesframe.reset_index()
#drop the old index
    lattesframe = lattesframe.drop('index', axis=1)

    if savefile:
        folder = os.getcwd()
        csvfile = "dataframe" + datetime.now().strftime('%Y%m%d%H%M%S') + \
            ".csv"

        lattesframe.to_csv(os.path.join(folder, csvfile),
                           index=False)

    return lattesframe

def lattes_classes_from_folder(folderlist, imin=2, imax=10, option=0, refdate1='',
                               refdate2=''):
    """
    From the Lattes CV files found in a given folder, extract some exploratory
    data results and perform analysis of the n mean publication profiles,
    where n varies from imin to imax.
    Args:
        folderlist: a list of strings containing names of the folders
        where the Lattes CV files are found.
        imin: lowest number of mean publication profiles analyzed.
        imax: highest number of mean publication profiles analyzed.
        option: argument that defines how the publication dates will be
            treated by the algorithm:
        option=0: production is based on calendar year. leave frame unchanged.
        option=1: first production value is first non-zero value of the
           production vector. The last vector indexes are substituted by zeros.
        option=2: first production year is first year of PIBIC scholarship.
           Only make sense to use it in PIBIC-based dataframes.
        refdate1: lowest date to limit the analysis.
        refdate2: highest date to limit the analysis.
    """
    import LattesLab as ll

    lattesframe = ll.get_dataframe_from_folders(folderlist, True)

    cleandata = lattesframe

    lattesframe = ll.filter_by_date(lattesframe, refdate1, refdate2)

    ll.lattes_age(lattesframe)

    ll.lattes_pibics(lattesframe)

    ll.degree_rate_year(lattesframe, 'M')

    ll.lattes_grad_level(lattesframe)

    pubdata = ll.get_pub_year_data(lattesframe)

    ll.first_nonzero(pubdata, lattesframe['anoPrimeiroPIBIC'], option)

    chartdata1, fig1 = ll.get_freq_pie_chart(lattesframe.nacionalidade,
                                             "Nationalities Frequency")

    chartdata2, fig2 = ll.get_freq_pie_chart(lattesframe.quantasVezesPIBIC,
                                             "PIBIC Scholarships Frequency")

    chartdata3, fig3 = \
        ll.get_ctgrs_pie_chart(lattesframe.atualizado,
                               "Frequency of Lattes Age in Days")

    cleandata = pubdata
    fpcs = []
    centers = []
    clusters = []

    print('Analysis with all data.')

    centers, clusters, fpcs = ll.set_fuzzycmeans_clstr(imin, imax, cleandata)

    #novo dataframe que recebe apenas os estudantes que publicaram
    print('Analysis with all researchers that have published at least once.')
    cleandata2 = cleandata[cleandata.sum(axis=1) != 0]
    fpcs2 = []
    centers2 = []
    clusters2 = []

    centers2, clusters2, fpcs2 = ll.set_fuzzycmeans_clstr(imin, imax, cleandata2)


    return lattesframe

def lattes_classes_from_frame(lattesframe, imin=2, imax=10, option=0):
    """
    From the Lattes CV dataframe previously processed, extract some exploratory
    data results and perform analysis of the n mean publication profiles,
    where n varies from imin to imax.
    Args:
        lattesframe: the Lattes CV dataframe where the Lattes CV data are found.
        imin: lowest number of mean publication profiles analyzed.
        imax: highest number of mean publication profiles analyzed.
    """
    import LattesLab as ll

    cleandata = lattesframe

    ll.lattes_age(lattesframe)

    ll.lattes_pibics(lattesframe)

    ll.degree_rate_year(lattesframe, 'M')

    ll.lattes_grad_level(lattesframe)

    pubdata = ll.get_pub_year_data(lattesframe)

    ll.first_nonzero(pubdata, lattesframe['anoPrimeiroPIBIC'], option)

    cleandata = pubdata
    fpcs = []
    centers = []
    clusters = []

    print('Analysis with all data.')

    centers, clusters, fpcs = ll.set_fuzzycmeans_clstr(imin, imax, cleandata)

    #novo dataframe que recebe apenas os estudantes que publicaram
    print('Analysis with all researchers that have published at least once.')
    cleandata2 = cleandata[cleandata.sum(axis=1) != 0]
    fpcs2 = []
    centers2 = []
    clusters2 = []

    centers2, clusters2, fpcs2 = ll.set_fuzzycmeans_clstr(imin, imax, cleandata2)

def filter_PIBICs(lattesframe, npibics=1):
    """Returns the rows of the researchers in arg lattesframe that have
    been part of the PIBIC scholarship at least once.
    Args:
        lattesframe: the pandas dataframe to be filtered.
        npibics: the minimum quantity of PIBICs to be filtered.
    """
    if npibics <= 0:
        print('Invalid arg npibics. Reverting to default npibics == 1.')
        npibics = 1

    return lattesframe.loc[lattesframe['quantasVezesPIBIC'] >= npibics]

def filter_by_phd_year(lattesframe, year0, year1):
    """Returns the rows of the researchers in arg lattesframe that have
    concluded their PhD program between args year0 and year1.
    Args:
        lattesframe: the pandas dataframe to be filtered.
        year0: the first year of the interval.
        year1: the last year of the interval.
    """
    if year0 > year1:
        dummy = year0
        year0 = year1
        year1 = dummy

    return lattesframe[(lattesframe['anoPrimeiroD'] >= year0) &
                       (lattesframe['anoPrimeiroD'] <= year1)]

def filter_by_lattes_age(lattesframe, agedays):
    """Returns the rows of the researchers in arg lattesframe that have
    a lattes age lower than the arg agemonths.
    Args:
        lattesframe: the pandas dataframe to be filtered.
        agedays: the maximum acceptable age of lattes CVs to be filtered
        from lattesframe. Measured in days.
    """
    from datetime import datetime

    mask = [(datetime.today() - datetime.strptime(i, '%d%m%Y')).days < agedays \
            for i in lattesframe['atualizado']]

    return lattesframe[mask]


def filter_by_date(lattesframe, refdate1='', refdate2=''):
    """Returns the rows of the researchers that have updated the Lattes CV
    before a certain date.
    Args:
        lattesframe: the pandas dataframe to be filtered.
        refdate1: the lower-bound limit date to be used as reference.
        String in the format ddmmyyyy.
        refdate2: the upper-bound limit date to be used as reference.
        String in the format ddmmyyyy.
    """

    from datetime import datetime

#verify usability of variable refdate2
    try:
        datetime.strptime(refdate2, '%d%m%Y')
    except:
        if refdate2.lower() == 'today':
            refdate2 = datetime.now().strftime('%d%m%Y')
        else:
            refdate2 = datetime.now().strftime('%d%m%Y')
            print('Upper-limit date invalid. Using default date of today.')

#verify usability of variable refdate1
    try:
        datetime.strptime(refdate1, '%d%m%Y')
    except:
        refdate1 = '01011900'
        print('Lower-limit date invalid. Using default date of 01/01/1900.')

#verify if the upper and lower limits are correctly defined
    if (datetime.strptime(refdate1, '%d%m%Y') >
            datetime.strptime(refdate2, '%d%m%Y')):
        dummie = refdate1
        refdate1 = refdate2
        refdate2 = dummie

    mask = [(datetime.strptime(refdate2, '%d%m%Y') > \
            datetime.strptime(i, '%d%m%Y')) & \
            (datetime.strptime(i, '%d%m%Y') > \
            datetime.strptime(refdate1, '%d%m%Y')) \
            for i in lattesframe['atualizado']]

    return lattesframe[mask]

def top_words_frame(folderlist, nwords=10):
    """Returns the a dataframe with the most important words found in the
    titles of articles and works per researcher.
    Args:
        cfolder: the folder where the Lattes CV files are found.
        nwords: quantity of words to be part of the dataframe.
    """
    import LattesLab as ll
    import zipfile
    import xml.etree.cElementTree as ET
    import pandas as pd

    columns = ["Name"] + ["Keyword" + str(i+1) for i in range(0, nwords)]

    wordsframe = pd.DataFrame(columns=columns)

#verify if the parameter folderlist is a list

    if not isinstance(folderlist, list):
        folderlist = [folderlist]

    terms = []

    [goodlist, badlist] = ll.get_files_list(folderlist)

    del badlist

    for filename in goodlist:
    #abre o arquivo zip baixado do site do lattes
        archive = zipfile.ZipFile(filename, 'r')
    #cvdata = archive.read('curriculo.xml')
        cvfile = archive.open('curriculo.xml', 'r')

    #get the summary information from lattes cv
        tree = ET.parse(cvfile)
        root = tree.getroot()

        dummy = root.findall('.//TRABALHOS-EM-EVENTOS')
        y1 = []
        if dummy != []:
            for x in dummy[0]:
                try:
                    y1.append(x[0].attrib['TITULO-DO-TRABALHO'])
                except:
                    pass

        dummy = root.findall('.//ARTIGOS-PUBLICADOS')
        y2 = []
        if dummy != []:
            for x in dummy[0]:
                try:
                    y2.append(x[0].attrib['TITULO-DO-ARTIGO'])
                except:
                    pass

    #   dummy = root.findall('.//CAPITULOS-DE-LIVROS-PUBLICADOS')
    #   y3 = []
    #   for x in dummy[0]:
    #        try:
    #           y3.append(x[0].attrib['TITULO-DO-CAPITULO-DO-LIVRO'])
    #        except:
    #            pass
    #   dummy = root.findall('.//LIVROS-PUBLICADOS-OU-ORGANIZADOS')
    #   y4 = []
    #   for x in dummy[0]:
    #       try:
    #           y4.append(x[0].attrib['TITULO-DO-LIVRO'])
    #        except:
    #            pass

        desc = y1 + y2
        if desc != []:
            topwords = ll.summary_list_top_words(desc, nwords, terms)
        else:
            topwords = ['']*nwords

        cvowner = root[0].attrib['NOME-COMPLETO']

        dummyframe = pd.DataFrame(data=[[cvowner] + topwords], columns=columns)

        wordsframe = wordsframe.append(dummyframe)

    return wordsframe

def is_in_df(name, df):
    """
        Finds if the researcher with name in arg name is present in the
        Lattes CV dataframe.
        Args:
            name: name of the researcher
            df: dataframe to be searched.
    """
    return not df.loc[lambda s: s == name].empty

def get_work_history_file(cfolder, filename, nyears):
    """
    Generates a vector of works found in a lattes CV.
    Args:
        cfolder: folder containing the researcher Lattes CV file.
        filename: name of the file containing the Lattes CV.
        nyears: number of years to be considered in the vector. It defines
        the dimension of the vector.
    """
    import xml.etree.ElementTree as ET
    import zipfile
    from datetime import datetime
    import os

    folder = os.path.normpath(cfolder)
    rightname = os.path.join(folder, filename)

    archive = zipfile.ZipFile(rightname, 'r')
    cvfile = archive.open(archive.namelist()[0], 'r')

    tree = ET.parse(cvfile)
    root = tree.getroot()

    #quantidade de trabalhos publicados
    x = root.findall('.//TRABALHOS-EM-EVENTOS')

    if not x:
        qtyworks = 0
    else:
        qtyworks = len(x[0].getchildren())

    nwork = []

    for i in range(0, qtyworks):
        nwork.append(x[0][i][0].attrib["ANO-DO-TRABALHO"])

    nworkyear = [0]*nyears

#num intervalo de NWORKS anos, contar a quantidade de publicacoes por ano
# de 2017 i=0 para tras

    for i in range(0, nyears):
        nworkyear[i] = nwork.count(str(datetime.now().year - i))

    return nworkyear

def get_paper_history_file(cfolder, filename, nyears):
    """
    Generates a vector of papers found in a lattes CV.
    Args:
        cfolder: folder containing the researcher Lattes CV file.
        filename: name of the file containing the Lattes CV.
        nyears: number of years to be considered in the vector. It defines
        the dimension of the vector.
    """
    import xml.etree.ElementTree as ET
    import zipfile
    from datetime import datetime
    import os

    folder = os.path.normpath(cfolder)
    rightname = os.path.join(folder, filename)

    archive = zipfile.ZipFile(rightname, 'r')
    cvfile = archive.open(archive.namelist()[0], 'r')

    tree = ET.parse(cvfile)
    root = tree.getroot()

    #quantidade de artigos publicados
    x = root.findall('.//ARTIGOS-PUBLICADOS')

    if len(x) > 0:
        npapers = len(x[0].getchildren())
    else:
        npapers = 0

    allpapers = []

    for i in range(0, npapers):
        allpapers.append(x[0][i][0].attrib["ANO-DO-ARTIGO"])

    npaperyear = [0]*nyears

#num intervalo de NWORKS anos, contar a quantidade de publicacoes por ano
# de 2017 i=0 para tras

    for i in range(0, nyears):
        npaperyear[i] = allpapers.count(str(datetime.now().year - i))

    return npaperyear

def get_history_frame_1(lattesframe, name, nyears):

    from datetime import datetime
    import matplotlib.pyplot as plt
    import numpy as np

    from pylab import rcParams
    rcParams['figure.figsize'] = 8, 6
    rcParams['figure.dpi'] = 96
    rcParams['font.size'] = 12

    dummy = [str(x) for x in lattesframe.columns if 'papers' in x]
    y1 = lattesframe.loc[lattesframe['Nome'] == name, dummy].values.tolist()[0]

    dummy = [str(x) for x in lattesframe.columns if 'works' in x]
    y2 = lattesframe.loc[lattesframe['Nome'] == name, dummy].values.tolist()[0]

    maxnyears = len(y2)

    x = [(datetime.now().year - x) for x in range(0, maxnyears)]

    if nyears > maxnyears:
        print('Number of years requested bigger than maximum supported ' + \
              'by dataframe. Using default value of ' + str(maxnyears) + '.')
    else:
        x = x[0:nyears]
        y1 = y1[0:nyears]
        y2 = y2[0:nyears]

    plt.figure(2)
    p2 = plt.bar(x, y2)
    p1 = plt.bar(x, y1, bottom=y2)
    plt.legend((p1[0], p2[0]), ('Papers', 'Works'))
    plt.xticks(np.arange(min(x), max(x), int((max(x)-min(x))/5)))
    plt.title('Publication history of \n' + name)
    plt.xlabel('Calendar year')
    plt.show()

    return [y1, y2]

def get_params(filename, param, nyears):
    """
    From the file in arg file, returns the quantity of documents correspondent
    to the parameters listed in the arg param. The function also returns the
    distribution of production of the types described in arg param per year,
    starting from the current year and moving a quantity of years to the past.
    This quantity is represented by arg nyears.
    Args:
        cfolder: folder where the Lattes CV file is found.
        filename: name of the file containing the Lattes CV.
        param: a list of strings containing the parameters to be retrieved from
        the Lattes CV file.
        Possible list of parameters:
            'papers' -> list of the published scientific articles.
            'works' -> list of works published in congresses.
            'orientations' -> list of people the researcher has oriented, both
            in masters and PhD programs.
            'chapters' -> list of chapters of books written.
            'books' -> list of books published.
        nyears: quantity of years to be analyzed
    """

    import xml.etree.ElementTree as ET
    import zipfile

    archive = zipfile.ZipFile(filename, 'r')
    cvfile = archive.open(archive.namelist()[0], 'r')
    tree = ET.parse(cvfile)
    root = tree.getroot()

    str_to_search = []
    n = []
    dist = []

    if not isinstance(param, list):
        param = [param]

    for iparam in param:
        ni = []
        xi = []
        str_to_search = []
        if iparam.lower() == 'papers':
            str_to_search.append('.//ARTIGO-PUBLICADO')

        elif iparam.lower() == 'works':
            str_to_search.append('.//TRABALHO-EM-EVENTOS')

        elif iparam.lower() == 'orientations':
            str_to_search.append('.//ORIENTACOES-CONCLUIDAS-PARA-MESTRADO')
            str_to_search.append('.//ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO')

        elif iparam.lower() == 'books':
            str_to_search.append('.//LIVRO-PUBLICADO-OU-ORGANIZADO')

        elif iparam.lower() == 'chapters':
            str_to_search.append('.//CAPITULO-DE-LIVRO-PUBLICADO')
        else:
            print('ERROR: INVALID PARAMETER CHOICE.')
            return [0, 0]

        for i in range(0, len(str_to_search)):
            xi.append(root.findall(str_to_search[i]))
            if len(xi[i]) > 0:
                ni.append(len(xi[i]))
            else:
                ni.append(0)

        yi = get_params_dist(xi, iparam, ni, nyears)
        dist.append(yi)
        n.append(ni)

    return [n, dist]

def get_params_dist(xmlset, param, n, nyears):
    """
    From the xml elements in arg xmlset, returns the sequence of production of
    the types described in arg param per year, starting from the current
    year and moving a quantity of years to the past. This quantity is
    represented by arg nyears.
    Args:
        xmlset: list of xml elements corresponding to the parameters to be
        analyzed.
        param: a string containing the parameter to be retrieved from
        the Lattes CV file.
        Possible parameters:
            'papers' -> list of the published scientific articles.
            'works' -> list of works published in congresses.
            'orientations' -> list of people the researcher has oriented, both
            in masters and PhD programs.
            'chapters' -> list of chapters of books written.
            'books' -> list of books published.
        n: list with the quantities of parameters to be retrieved from the
        xml document.
        nyears: quantity of years to be analyzed
    """
    import xml.etree.ElementTree as ET
    from datetime import datetime

    yearseq = []

    for i in range(0, len(n)):
        yearseq.append([])
        for j in range(0, n[i]):
            if param.lower() == 'papers':
                yearseq[i].append(xmlset[i][j][0].attrib['ANO-DO-ARTIGO'])

            elif param.lower() == 'works':
                yearseq[i].append(xmlset[i][j][0].attrib['ANO-DO-TRABALHO'])

            elif param.lower() == 'orientations':
                yearseq[i].append(xmlset[i][j][0].attrib['ANO'])

            elif param.lower() == 'books':
                yearseq[i].append(xmlset[i][j][0].attrib['ANO'])

            elif param.lower() == 'chapters':
                yearseq[i].append(xmlset[i][j][0].attrib['ANO'])
            else:
                print('ERROR: INVALID PARAMETER CHOICE.')
                return
    dist = []
    for i in range(0, len(n)):
        dist.append([0]*nyears)
        for j in range(0, nyears):
            dist[i][j] = yearseq[i].count(str(datetime.now().year - j))

    return dist

def get_param_history_file(rightname, param, nyears):
    """
    From the file in arg filename, returns the sequence of production of
    the types described in arg param per year, starting from the current
    year and moving a quantity of years to the past. This quantity is
    represented by arg nyears.
    Args:
        cfolder: folder where the Lattes CV file is found.
        filename: name of the file containing the Lattes CV.
        param: a list of strings containing the parameters to be retrieved from
        the Lattes CV file.
        Possible list of parameters:
            'papers' -> list of the published scientific articles.
            'works' -> list of works published in congresses.
            'orientations' -> list of people the researcher has oriented, both
            in masters and PhD programs.
            'chapters' -> list of chapters of books written.
            'books' -> list of books published.
        nyears: quantity of years to be analyzed
    """
    [x, y] = get_params(rightname, param, nyears)

    return y

def get_history_file_1(cfolder, filename, param, nyears):
    """
    Exhibits a graphic with the production of the researcher. The 'production'
    is defined according to the parameters inserted.

    Args:
        cfolder: the folder where the Lattes CV file is found
        filename: the name of the Lattes CV file downloaded from the Lattes CV
        website.
        param: a list of parameters to be retrieved from Lattes CV file.
        Possible list of parameters:
            'papers' -> list of the published scientific articles.
            'works' -> list of works published in congresses.
            'orientations' -> list of people the researcher has oriented, both
            in masters and PhD programs.
            'chapters' -> list of chapters of books written.
            'books' -> list of books published.
        nyears: quantity of years to be analyzed
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import LattesLab as ll
    from datetime import datetime
    import numpy as np
    import os

    from pylab import rcParams
    rcParams['figure.figsize'] = 8, 6
    rcParams['figure.dpi'] = 96
    rcParams['font.size'] = 12

    if not isinstance(param, list):
        param = [param]

    folder = os.path.normpath(cfolder)
    rightfile = os.path.join(folder, filename)

    y = get_param_history_file(rightfile, param, NWORKS)

    x = [(datetime.now().year - x) for x in range(0, NWORKS)]

    if nyears > NWORKS:
        print('Number of years requested bigger than maximum supported ' + \
              'by dataframe. Using default value of ' + str(NWORKS) + '.')
    else:
        x = x[0:nyears]
        for i in y:
            for j in i:
                j = j[0:nyears]

    yplot = []
    for i in y:
        for j in i:
            yplot.append(j)

    fig, ax = plt.subplots()

    p = []
    ybottom = [0]*nyears
    for i in range(0, len(yplot)):
        dummy = plt.bar(x, yplot[i], bottom=ybottom)
        ybottom = list(np.add(ybottom, yplot[i]))
        p.append(dummy)

    xlegend = []
    for i in param:
        if i == 'orientations':
            xlegend.append('orientations_master')
            xlegend.append('orientations_phd')
        else:
            xlegend.append(i)

    plt.yticks(np.arange(0, max(ybottom)*1.1 + 5, 5))
    plt.xticks(np.arange(min(x), max(x), 5))
    plt.grid(True)

    name = ll.lattes_owner(cfolder, filename)
    plt.title('Publication history of \n' + name)
    
    plt.xlabel("Calendar year")
    plt.ylabel("Quantity of publication items")

    ax.legend(p[::-1], xlegend[::-1], loc=2, 
              bbox_to_anchor=(1.05, 1), borderaxespad=0.,
              shadow=True)

    plt.show()

    return yplot

def get_history_file_n(folderlist, param, nyears):
    """
    Exhibits a graphic with the production of the group of researchers.
    The 'production' is defined according to the parameters inserted.

    Args:
        folderlist: the list of folders where the Lattes CV file are
        found.
        param: a list of parameters to be retrieved from Lattes CV file.
        Possible list of parameters:
            'papers' -> list of the published scientific articles.
            'works' -> list of works published in congresses.
            'orientations' -> list of people the researcher has oriented, both
            in masters and PhD programs.
            'chapters' -> list of chapters of books written.
            'books' -> list of books published.
        nyears: quantity of years to be analyzed
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from datetime import datetime
    import numpy as np

    from pylab import rcParams
    rcParams['figure.figsize'] = 8, 6
    rcParams['figure.dpi'] = 96
    rcParams['font.size'] = 12

    if not isinstance(folderlist, list):
        folderlist = [folderlist]

#filters the zip files

    ziplist = nonziplist = []
    [ziplist, nonziplist] = get_files_list(folderlist)

    if not isinstance(param, list):
        param = [param]

    count = 0

    for rightname in ziplist:
        count += 1

        dummy = get_param_history_file(rightname, param, NWORKS)
        if count == 1:
            y = dummy
        else:
            y = sumlist(y, dummy)

    x = [(datetime.now().year - x) for x in range(0, NWORKS)]

    if nyears > NWORKS:
        print('Number of years requested bigger than maximum supported ' + \
              'by dataframe. Using default value of ' + str(NWORKS) + '.')
    else:
        x = x[0:nyears]
        for i in y:
            for j in i:
                j = j[0:nyears]

    yplot = []
    for i in y:
        for j in i:
            yplot.append(j)

    fig, ax = plt.subplots()

    p = []
    ybottom = [0]*nyears
    for i in range(0, len(yplot)):
        dummy = plt.bar(x, yplot[i], bottom=ybottom)
        ybottom = list(np.add(ybottom, yplot[i]))
        p.append(dummy)

    xlegend = []
    for i in param:
        if i == 'orientations':
            xlegend.append('orientations_master')
            xlegend.append('orientations_phd')
        else:
            xlegend.append(i)

    plt.yticks(np.arange(0, max(ybottom)*1.1, round((max(ybottom)+5)/100)*10))
    plt.xticks(np.arange(min(x), max(x), 5))
    plt.grid(True)

    plt.title('Group Publication History')

    plt.xlabel("Calendar year")
    plt.ylabel("Quantity of publication items")

    ax.legend(p[::-1], xlegend[::-1], loc=2, 
              bbox_to_anchor=(1.05, 1), borderaxespad=0.,
              shadow=True)

    plt.show()

    return yplot


def recode_str(str_input):
    """
    Recodes the input string in the UTF-8 coding.
    Args:
        str_input: the string to be decoded.
    """

    dummy = str(str_input).encode(encoding='ISO-8859-1', errors='strict'). \
        decode(encoding='utf-8', errors='ignore')

    return dummy

def get_pub_dataframe_from_folders(folderlist, savefile=True):
    """
    From the Lattes CVs in a folder, build a dataframe based on the title
    of the works produced by each researcher.
    Args:
        folderlist: name of the folder that contains the Lattes CV. The
            Lattes CV files are downloaded as .zip files containing a
            .xml file.
        savefile: if True, the dataframe is stored in a .csv file for
            posterior use.
    """
    from datetime import datetime
    import pandas as pd
    import xml.etree.ElementTree as ET
    import zipfile
    import os

#initiate the dataframe
    columns = ['Nome',
               'lattesId',
               'type',
               'title',
               'year',
               'idiom']

#verify if the parameter folderlist is a list

    if not isinstance(folderlist, list):
        folderlist = [folderlist]

#filters the zip files

    ziplist = nonziplist = []
    [ziplist, nonziplist] = get_files_list(folderlist)

    count = 0
    pubframe = pd.DataFrame(columns=columns)

    for rightname in ziplist:
        count += 1
        archive = zipfile.ZipFile(rightname, 'r')
        cvfile = archive.open(archive.namelist()[0], 'r')

        tree = ET.parse(cvfile)
        root = tree.getroot()

    #Retrieve genaral data
        name = root[0].attrib["NOME-COMPLETO"]
        try:
            readid = str(root.attrib["NUMERO-IDENTIFICADOR"])
        except:
            readid = str(9999999999999999)

        #WORKS PUBLISHED
        x = root.findall('.//TRABALHOS-EM-EVENTOS')
        try:
            for y in x[0]:
                xtype = 'work'
                try:
                    xtitle = y[0].attrib["TITULO-DO-TRABALHO"]
                except:
                    xtitle = "TITLE_NOT_FOUND"
                try:
                    xyear = y[0].attrib["ANO-DO-TRABALHO"]
                except:
                    xyear = "9999"
                try:
                    xidiom = y[0].attrib["IDIOMA"]
                except:
                    xidiom = "IDIOM_NOT_FOUND"
    
                dummy = [name, readid, xtype, xtitle, xyear, xidiom]
    
                dummy2 = pd.DataFrame(data=[dummy], columns=columns)
    
                pubframe = pubframe.append(dummy2)
        except:
            pass

        x = root.findall('.//ARTIGO-PUBLICADO')
        try:
            for y in x:
                xtype = 'paper'
                try:
                    xtitle = y[0].attrib["TITULO-DO-ARTIGO"]
                except:
                    xtitle = "TITLE_NOT_FOUND"
                try:
                    xyear = y[0].attrib["ANO-DO-ARTIGO"]
                except:
                    xyear = "9999"
                try:
                    xidiom = y[0].attrib["IDIOMA"]
                except:
                    xidiom = "IDIOM_NOT_FOUND"
                dummy = [name, readid, xtype, xtitle, xyear, xidiom]
    
                dummy2 = pd.DataFrame(data=[dummy], columns=columns)
    
                pubframe = pubframe.append(dummy2)
        except:
            pass

        x = root.findall('.//LIVRO-PUBLICADO-OU-ORGANIZADO')
        try:
            for y in x:
                xtype = 'book'
                try:
                    xtitle = y[0].attrib["TITULO-DO-LIVRO"]
                except:
                    xtitle = "TITLE_NOT_FOUND"
                try:
                    xyear = y[0].attrib["ANO"]
                except:
                    xyear = "9999"
                try:
                    xidiom = y[0].attrib["IDIOMA"]
                except:
                    xidiom = "IDIOM_NOT_FOUND"
    
                dummy = [name, readid, xtype, xtitle, xyear, xidiom]
    
                dummy2 = pd.DataFrame(data=[dummy], columns=columns)
    
                pubframe = pubframe.append(dummy2)
        except:
            pass

        x = root.findall('.//CAPITULO-DE-LIVRO-PUBLICADO')
        try:
            for y in x:
                xtype = 'chapter'
                try:
                    xtitle = y[0].attrib["TITULO-DO-CAPITULO-DO-LIVRO"]
                except:
                    xtitle = "TITLE_NOT_FOUND"
                try:
                    xyear = y[0].attrib["ANO"]
                except:
                    xyear = "9999"
                try:
                    xidiom = y[0].attrib["IDIOMA"]
                except:
                    xidiom = "IDIOM_NOT_FOUND"
    
                dummy = [name, readid, xtype, xtitle, xyear, xidiom]
    
                dummy2 = pd.DataFrame(data=[dummy], columns=columns)
    
                pubframe = pubframe.append(dummy2)
        except:
            pass

#reindex the dataframe
    pubframe = pubframe.reset_index()
#drop the old index
    pubframe = pubframe.drop('index', axis=1)

    if savefile:
        folder = os.getcwd()
        csvfile = "dataframe" + datetime.now().strftime('%Y%m%d%H%M%S') + \
            ".csv"

        pubframe.to_csv(os.path.join(folder, csvfile),
                        index=False)
    return pubframe

def sumlist(x, y):
    """
    Sums two lists of the same shape elementwise. Returns the sum.
    """
    z = x
    for i in range(0, len(z)):
        if isinstance(x[i], list) and isinstance(y[i], list):
            z[i] = sumlist(x[i], y[i])
        else:
            z[i] = x[i] + y[i]
    return z

def list_of_list_to_list(listoflist):
    """
    Transforms a list of lists in a single list.
    params:
        listoflist: the list of lists
    """
    singlelist = []
    for xlist in listoflist:
        for xelem in xlist:
            singlelist.append(xelem)

    return singlelist

def get_wordclouds_from_pub_frame(pubframe, saveit=False):
    """
    From the publications found in the dataframe pubframe, generate
    a wordcloud referring to each researcher.
    params:
        pubframe(dataframe): the dataframe containing the publication
        data
        saveit(boolean): if True, saves snapshots of the wordclouds to
        the working dir.
    """
    import os

    folder = os.getcwd()

    allnames = pubframe.Nome.unique().tolist()

    alltitles = []
    topwords = []

    for name in allnames:
        dummyframe = pubframe.loc[pubframe.Nome == name]
        titles = []
        for i in range(0, len(dummyframe)):
            titles.append(dummyframe.iloc[i].title)
        alltitles.append(titles)
        top50 = summary_list_top_words(titles, 50)
        if top50:
            topwords.append(top50)
            word_list_to_cloud(folder, top50, 'WORDCLOUD: \n' + name,
                               saveit)
        else:
            pass

    dummytitles = list_of_list_to_list(alltitles)
    top50 = summary_list_top_words(dummytitles, 50)
    topwords.append(top50)
    word_list_to_cloud(folder, top50, 'WORDCLOUD: ALL', saveit)

    return allnames, alltitles, topwords

def get_grad_count(root, gradtype):
    """
        Finds in the Lattes CV parsed in arg root the year the first 
        graduation of the type described in arg gradtype, as well as the 
        quantity of graduations of the same type.
        Args:
            root: the parsed Lattes CV file
            gradtype: type of graduation. Can assume the following values:
                "GRADUACAO", "MESTRADO", "DOUTORADO" or , "POS-DOUTORADO":
    """
    
    if gradtype not in ["GRADUACAO", "MESTRADO", "DOUTORADO", 
                        "POS-DOUTORADO", "LIVRE-DOCENCIA"]:
        print('Graduation type not recognized. Using default value of ' + \
              '"GRADUACAO".')
        gradtype = "GRADUACAO"
    
    ngrad = ano1grad = 0

    x = root.findall('.//FORMACAO-ACADEMICA-TITULACAO')

    if x != []:
        for i in range(0, len(x[0].getchildren())):
            if x[0][i].tag == gradtype:
                [ano1grad, ngrad] = get_grad_years(x[0][i], x[0][i].tag)
    
    return [ano1grad, ngrad]