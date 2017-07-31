# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 11:16:32 2017

@author: thiag
"""

import xml.etree.ElementTree as ET
import zipfile
import pandas as pd
import os


columns = ['Nome',
           'lattesId',
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
           ["works" + str(2017 - i) for i in range(0, 20)] + \
           ["papers" + str(2017 - i) for i in range(0, 20)]
    
lattesframe = pd.DataFrame(columns=columns)


folder= "D:\\thiag\\Documents\\INPE\\Research\\Datasets\\" +  \
        "DoutoresEngenharias\\Eng1\\"
        
folder = "C:\\Users\\thiag\\Desktop\\1\\"
        
fileslist = os.listdir(folder)

for filename in fileslist:
    if not filename.endswith('.zip'):
        fileslist.remove(filename)

#criar uma lista dos arquivos com erro
#fazer um teste com um lattes vazio

nresearchers = len(fileslist)
count = 0

for filename in fileslist:
    
    count += 1
    
    archive = zipfile.ZipFile((folder + filename), 'r')
    if (archive.namelist()[0][-3:]=='xml')|(archive.namelist()[0][-3:]=='XML'):
        cvfile = archive.open(archive.namelist()[0], 'r')
    else:
        print('Error: file ' + archive.namelist()[0] + 'is not a xml file.')
        break


    tree = ET.parse(cvfile)
    root = tree.getroot()
    result = ''

#lista todos os atributos do arquivo XML
    elemtree = []

    for elem in tree.iter():
        elemtree.append(elem.tag)
    

    elemtree = list(set(elemtree))

    root.tag
    root.attrib
    root.getchildren()


#Dados gerais
    readid = str(root.attrib["NUMERO-IDENTIFICADOR"])
    lastupd = str(root.attrib["DATA-ATUALIZACAO"])
    name = root[0].attrib["NOME-COMPLETO"]

#Dados de formacao academica

    ngrad = nmaster = nphd = nposdoc = 0
    ano1grad = ano1master = ano1phd = ano1postdoc = 0

    x = root.findall('.//FORMACAO-ACADEMICA-TITULACAO')
	
    for i in range(0, len(x[0].getchildren())):
        if x[0][i].tag == "GRADUACAO":
#Curso foi concluido?
#        if root[0][3][i].attrib["STATUS-DO-CURSO"]=="CONCLUIDO":
            if (x[0][i].attrib["ANO-DE-CONCLUSAO"]!="")| \
            (x[0][i].attrib["ANO-DE-INICIO"]!=""):
                ngrad = ngrad + 1
                if ngrad==1:
                    if x[0][i].attrib["ANO-DE-CONCLUSAO"]=="":
                        x[0][i].attrib["ANO-DE-CONCLUSAO"] = \
                        str(int(x[0][i].attrib["ANO-DE-INICIO"]) + 4)
                    ano1grad=int(x[0][i].attrib["ANO-DE-CONCLUSAO"])
                    
#Se terminou o curso
                if x[0][i].attrib["ANO-DE-CONCLUSAO"]=="":
                    x[0][i].attrib["ANO-DE-CONCLUSAO"] = \
                    str(int(x[0][i].attrib["ANO-DE-INICIO"]) + 4)
                if ano1grad > int(x[0][i].attrib["ANO-DE-CONCLUSAO"]):
                    ano1grad = int(x[0][i].attrib["ANO-DE-CONCLUSAO"])

        elif x[0][i].tag == "MESTRADO":
#Curso foi concluido?
#        if root[0][3][i].attrib["STATUS-DO-CURSO"]=="CONCLUIDO":
            if (x[0][i].attrib["ANO-DE-CONCLUSAO"]!="")| \
            (x[0][i].attrib["ANO-DE-INICIO"]!=""):
                nmaster = nmaster + 1
                if nmaster==1:
                    if x[0][i].attrib["ANO-DE-CONCLUSAO"]=="":
                        x[0][i].attrib["ANO-DE-CONCLUSAO"] = \
                        str(int(x[0][i].attrib["ANO-DE-INICIO"]) + 2)    
                    ano1master = int(x[0][i].attrib["ANO-DE-CONCLUSAO"])
                
                if x[0][i].attrib["ANO-DE-CONCLUSAO"]=="":
                    x[0][i].attrib["ANO-DE-CONCLUSAO"] = \
                    str(int(x[0][i].attrib["ANO-DE-INICIO"]) + 2)    
                if ano1master > int(x[0][i].attrib["ANO-DE-CONCLUSAO"]):
                    ano1master = int(x[0][i].attrib["ANO-DE-CONCLUSAO"])
            
        elif x[0][i].tag == "DOUTORADO":
            if (x[0][i].attrib["ANO-DE-CONCLUSAO"]!="")| \
            (x[0][i].attrib["ANO-DE-INICIO"]!=""):
                nphd = nphd + 1
                if nphd==1:
                    if x[0][i].attrib["ANO-DE-CONCLUSAO"]=="":
                        x[0][i].attrib["ANO-DE-CONCLUSAO"] = \
                        str(int(x[0][i].attrib["ANO-DE-INICIO"]) + 4)
                    ano1phd=int(x[0][i].attrib["ANO-DE-CONCLUSAO"])
#Se terminou o curso
                if x[0][i].attrib["ANO-DE-CONCLUSAO"]=="":
                    x[0][i].attrib["ANO-DE-CONCLUSAO"] = \
                    str(int(x[0][i].attrib["ANO-DE-INICIO"]) + 4)
                
                if ano1phd > int(x[0][i].attrib["ANO-DE-CONCLUSAO"]):
                    ano1phd = int(x[0][i].attrib["ANO-DE-CONCLUSAO"])        
        
        elif x[0][i].tag == "POS-DOUTORADO":
            if (x[0][i].attrib["ANO-DE-CONCLUSAO"]!="")| \
            (x[0][i].attrib["ANO-DE-INICIO"]!=""):
                nposdoc = nposdoc + 1
                if nposdoc==1:
                    if x[0][i].attrib["ANO-DE-CONCLUSAO"]=="":
                        x[0][i].attrib["ANO-DE-CONCLUSAO"] = \
                        str(int(x[0][i].attrib["ANO-DE-INICIO"]) + 2)
                    ano1postdoc=int(x[0][i].attrib["ANO-DE-CONCLUSAO"])
#Se terminou o curso
                if x[0][i].attrib["ANO-DE-CONCLUSAO"]=="":
                    x[0][i].attrib["ANO-DE-CONCLUSAO"] = \
                    str(int(x[0][i].attrib["ANO-DE-INICIO"]) + 2)
                
                if ano1postdoc > int(x[0][i].attrib["ANO-DE-CONCLUSAO"]):
                    ano1postdoc = int(x[0][i].attrib["ANO-DE-CONCLUSAO"])        

#producao bibliografica

    root[1].getchildren()

#quantidade de trabalhos publicados
    x = root.findall('.//TRABALHOS-EM-EVENTOS')
    
    if not x: 
        nworks = 0
    else:
        nworks = len(x[0].getchildren())
    
    nprod = []

    for i in range(0, nworks):
        nprod.append(x[0][i][0].attrib["ANO-DO-TRABALHO"])
    
    nprodyear = [0]*20

#num intervalo de 20 anos, contar a quantidade de publicacoes por ano de 2017 
#i=0 para tras

    for i in range(0, len(nprodyear)):
        nprodyear[i] = nprod.count(str(2017 - i))
    
#quantidade de artigos publicados
    x = root.findall('.//ARTIGOS-PUBLICADOS')

    if len(x) > 0:
        npapers = len(x[0].getchildren())
    else:
        npapers = 0
    
    allpapers = []

    for i in range(0, npapers):
        allpapers.append(x[0][i][0].attrib["ANO-DO-ARTIGO"])
    
    npaperyear = [0]*20

#num intervalo de 20 anos, contar a quantidade de publicacoes por ano de 2017 
#i=0 para tras

    for i in range(0, len(npaperyear)):
        npaperyear[i] = allpapers.count(str(2017 - i))
    
#Procurando pela quantidade de iniciações cientificas
#    root[0][4][0].tag #estava usando esse, mudei para o debaixo
    x = root.findall('.//*[@OUTRO-VINCULO-INFORMADO="Iniciação Cientifica"]') + \
    root.findall('.//*[@OUTRO-VINCULO-INFORMADO="Iniciação Científica"]') + \
    root.findall('.//*[@OUTRO-ENQUADRAMENTO-FUNCIONAL-INFORMADO=' +  \
                       '"Aluno de Iniciação Cientifica"]') + \
    root.findall('.//*[@OUTRO-ENQUADRAMENTO-FUNCIONAL-INFORMADO=' +  \
                       '"Aluno de Iniciação Científica"]')

    qtdeanos = 0
    if not x:
        ano1PIBIC = 0
    else:
        ano1PIBIC = int(x[0].attrib["ANO-INICIO"]) #era 9999
    
    for elem in x:
        if (elem.attrib["ANO-FIM"]==""): # & \
#            (elem.attrib["SITUACAO"]=="EM_ANDAMENTO"):

            elem.attrib["ANO-FIM"] = "2017"
        if elem.tag == "VINCULOS":
            if ((elem.attrib["MES-INICIO"]!="")&(elem.attrib["MES-FIM"]!="")):
                qtdeanos += (float(elem.attrib["ANO-FIM"]) - 
    				   			  float(elem.attrib["ANO-INICIO"]) +
					   		     (float(elem.attrib["MES-FIM"]) - 
						   	     float(elem.attrib["MES-INICIO"]))/12)
            else:
#As vezes o projeto comeca e termina no mesmo ano. Corrigir o delta anos.            
                if elem.attrib["ANO-FIM"] == elem.attrib["ANO-INICIO"]:
                    qtdeanos += 1
                else:
                    qtdeanos += (float(elem.attrib["ANO-FIM"]) -
                                 float(elem.attrib["ANO-INICIO"]))
        if ano1PIBIC > int(elem.attrib["ANO-INICIO"]):
            ano1PIBIC = int(elem.attrib["ANO-INICIO"])

    qtdePIBIC = round(qtdeanos)
    
    x = [name, readid, lastupd, qtdePIBIC, ano1PIBIC,
         ngrad, ano1grad ,nmaster, ano1master, nphd,
         ano1phd, nposdoc, ano1postdoc] + nprodyear + npaperyear
    
    dummie = pd.DataFrame(data=[x], columns=columns)
         
    lattesframe = lattesframe.append(dummie)
    
    print(str(100*count/len(fileslist)) + '%%')



lattesframe.to_csv(folder + 'dataframe.csv', index=False)