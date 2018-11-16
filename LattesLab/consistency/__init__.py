# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 20:12:50 2018

@author: thiag
"""

from pylab import rcParams
rcParams['figure.figsize'] = 8, 6
rcParams['figure.dpi'] = 96
rcParams['font.size'] = 22

###--------------------------------------------------------------------###
# Consistencies functions go below

def test_summary(root, printerror=False):
    """Verifies if there are problems with the Lattes CV summary parsed
    in the argument root.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, there is a summary. If 1, the summary is absent.
        printerror: if True, shows the errors in the command window.
    """
    summaryflag = 0

#get the summary information from lattes cv
    try:
        desc = root[0][0].attrib['TEXTO-RESUMO-CV-RH']
    except:
        desc = ""
        if printerror: print('Resumo não encontrado.')
        summaryflag = 1

    return summaryflag

def test_id(root, printerror=False):
    """Verifies if there are problems with the Lattes CV ID parsed in the argument root.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree language
        flag: if 0, there is an ID. If 1, the ID is absent.
        printerror: if True, shows the errors in the command window.
    """

    idflag = 0

#get the ID information from lattes cv
    try:
        readid = str(root.attrib["NUMERO-IDENTIFICADOR"])
    except:
        readid = str(9999999999999999)
        if printerror: print('Numero identificador não encontrado.')
        idflag = 1

    return idflag

def test_email(root, printerror=False):
    """Verifies if there are problems with the Lattes CV email parsed in the argument root.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree language
        flag: if 0, there is an email. If 1, the email is absent.
        printerror: if True, shows the errors in the command window.
    """

    emailflag = 0

#consistencia do email
    try:
        email = root[0][2][0].attrib["E-MAIL"]
    except:
        email = ''
        if printerror: print('E-mail não encontrado.')
        emailflag = 1

    return emailflag

def test_update(root, refdate='', printerror=False):
    """Verifies when the Lattes CV email parsed in the argument root was
    last updated.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        refdate: if passed, is the date used as reference for the age of
        the Lattes CVs.
        days: number of days since last Lattes CV update.
        printerror: if True, shows the errors in the command window.
    """
    from datetime import datetime

    try:
        datetime.strptime(refdate, '%d%m%Y')
    except:
        if refdate.lower() == 'today':
            refdate = datetime.now().strftime('%d%m%Y')
        else:
            refdate = datetime.now().strftime('%d%m%Y')
            if printerror: print('Reference date invalid. Using default date of today.')

#consistencia da data de atualização.

    lastupd = str(root.attrib["DATA-ATUALIZACAO"])
    dayslate = (datetime.today() - datetime.strptime(lastupd, '%d%m%Y')).days
    if printerror: print('Ultima atualizacao há ' + str(dayslate) + ' dias.')
    return dayslate

def test_language(root, printerror=False):
    """Verifies if there are problems with the Lattes CV languages parsed
    in the argument root.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, there are languages defined in the Lattes CV. If
        1, no languages were defined.
        printerror: if True, shows the errors in the command window.
    """

    languageflag = 0

#consistencia de idiomas.
    try:
        x = root.findall('.//IDIOMA')
    except:
        x = ''
        if printerror: print('Idiomas não definidos.')
        languageflag = 1

    return languageflag

def test_nationality(root, printerror=False):
    """Verifies if there are problems with the nationalities in the
    Lattes CV parsed in the argument root.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, there are languages defined in the Lattes CV.
        If 1, no languages were defined.
        printerror: if True, shows the errors in the command window.
    """
    nationflag = 0

#consistencia da nacionalidade.
    try:
        nation = root[0].attrib["SIGLA-PAIS-NACIONALIDADE"]
    except:
        nation = "Unspecified"
        if printerror: print('Nacionalidade não definida.')
        nationflag = 1

    return nationflag

def test_grad_end(root, gradtype, printerror=False):
    """Verifies if the graduations cited in the Lattes CV parsed in
    the argument root were concluded.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        gradtype: type of graduation. Can assume the following values:
            "GRADUACAO", "MESTRADO", "DOUTORADO" or , "POS-DOUTORADO":
        flag: if 0, the graduations of the type gradtype were
        concluded. If 1, they were not.
        printerror: if True, shows the errors in the command window.
    """
    gradconclflag = 0

    if gradtype not in ["GRADUACAO", "MESTRADO", "DOUTORADO",
                        "POS-DOUTORADO"]:
        if printerror: print('Tipo de graduação inválido. Revertendo ' + \
                             'para o valor padrão "GRADUACAO".')
        gradtype = "GRADUACAO"

#Consistencia dos dados academicos

    x = root.findall('.//FORMACAO-ACADEMICA-TITULACAO')

    if x != []:
        for i in range(0, len(x[0].getchildren())):
            try:
                year0 = x[0][i].attrib["ANO-DE-INICIO"]
            except:
                year0 = ""
            try:
                year1 = x[0][i].attrib["ANO-DE-CONCLUSAO"]
            except:
                year1 = ""

            if year0 != "" and year1 == "":
                if gradtype == "GRADUACAO":
                    if printerror: print('Graduacao nao concluida.')
                    gradconclflag = 1
                elif gradtype == "MESTRADO":
                    if printerror: print('Mestrado nao concluido.')
                    gradconclflag = 1
                elif gradtype == "DOUTORADO":
                    if printerror: print('Doutorado nao concluido.')
                    gradconclflag = 1
                elif gradtype == "POS-DOUTORADO":
                    if printerror: print('Pos-Doutorado nao concluido.')
                    gradconclflag = 1
    else:
        gradconclflag = 'N/A'

    return gradconclflag

def test_grad_seq(root, printerror=False):
    """Verifies if the graduations cited in the Lattes CV parsed in
    the argument root are in the expected sequence, and that there
    aren't graduations 'missing'.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, the graduations are in the expected sequence.
        If 1, they were not.
        printerror: if True, shows the errors in the command window.
    """
    import LattesLab as ll

    gradseqflag = 0

#consistencia da ordem de graduacoes
    ngrad = nmaster = nphd = nposdoc = nlivredoc = 0
    ano1grad = ano1master = ano1phd = ano1postdoc = ano1livredoc = 0

    x = root.findall('.//FORMACAO-ACADEMICA-TITULACAO')
    if x != []:
        for i in range(0, len(x[0].getchildren())):
            if x[0][i].tag == "GRADUACAO":
                [ano1grad, ngrad] = \
                    ll.get_grad_years(x[0][i], x[0][i].tag)
            elif x[0][i].tag == "MESTRADO":
                [ano1master, nmaster] = \
                    ll.get_grad_years(x[0][i], x[0][i].tag)
            elif x[0][i].tag == "DOUTORADO":
                [ano1phd, nphd] = \
                    ll.get_grad_years(x[0][i], x[0][i].tag)
            elif x[0][i].tag == "POS-DOUTORADO":
                [ano1postdoc, nposdoc] = \
                    ll.get_grad_years(x[0][i], x[0][i].tag)
            elif x[0][i].tag == "LIVRE-DOCENCIA":
                [ano1livredoc, nlivredoc] = \
                    ll.get_grad_years(x[0][i], x[0][i].tag)

    if ngrad == 0 and nmaster + nphd + nposdoc + nlivredoc > 0:
        if printerror: print('Graduacao nao detectada, pos graduacao ' + \
                             'ou livre-docencia detectada.')
        gradseqflag = 1
    elif nmaster == 0 and nphd + nposdoc > 0:
        if printerror: print('Mestrado nao detectado, doutorado detectado.')
        gradseqflag = 1
    elif nphd == 0 and nposdoc > 0:
        if printerror: print('Doutorado nao detectado, pos-doutorado ' + \
                             'detectado.')
        gradseqflag = 1

    return gradseqflag

def grad_level(root, printerror=False):
    """Verifies the highest graduation level cited in the Lattes CV.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        gradlevel: the highest graduation level cited in the parsed Lattes
        CV.
        printerror: if True, shows the errors in the command window.
    """
    import LattesLab as ll

    gradlevel = "NÃO-INFORMADO"

#consistencia da ordem de graduacoes
    ngrad = nmaster = nphd = nposdoc = nlivredoc = 0
    ano1grad = ano1master = ano1phd = ano1postdoc = ano1livredoc = 0

    x = root.findall('.//FORMACAO-ACADEMICA-TITULACAO')
    if x != []:
        for i in range(0, len(x[0].getchildren())):
            if x[0][i].tag == "GRADUACAO":
                [ano1grad, ngrad] = \
                    ll.get_grad_years(x[0][i], x[0][i].tag)
            elif x[0][i].tag == "MESTRADO":
                [ano1master, nmaster] = \
                    ll.get_grad_years(x[0][i], x[0][i].tag)
            elif x[0][i].tag == "DOUTORADO":
                [ano1phd, nphd] = \
                    ll.get_grad_years(x[0][i], x[0][i].tag)
            elif x[0][i].tag == "POS-DOUTORADO":
                [ano1postdoc, nposdoc] = \
                    ll.get_grad_years(x[0][i], x[0][i].tag)
            elif x[0][i].tag == "LIVRE-DOCENCIA":
                [ano1livredoc, nlivredoc] = \
                    ll.get_grad_years(x[0][i], x[0][i].tag)

        if nposdoc > 0:
            gradlevel = "POS-DOUTORADO"
        elif nphd > 0:
            gradlevel = "DOUTORADO"
        elif nmaster > 0:
            gradlevel = "MESTRADO"
        elif ngrad > 0:
            gradlevel = "GRADUACAO"

    return gradlevel

def test_if_work(root, printerror=False):
    """Verifies if the Lattes CV parsed in the argument root contain
    published works.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, the Lattes CV declares that the resercher produced
        works. If 1, the researcher hasn't declared that they
        presented works in events.
        printerror: if True, shows the errors in the command window.
    """
    gradnworksflag = 0

#consistencia do numero de trabalhos
    x = root.findall('.//TRABALHOS-EM-EVENTOS')

    if not x:
        qtyworks = 0
        if printerror: print('Nenhum trabalho publicado.')
        gradnworksflag = 1
    else:
        qtyworks = len(x[0].getchildren())

    return gradnworksflag

#Consistencia da existencia de DOI em todos os trabalhos

def test_DOI_work(root, printerror=False):
    """Verifies if the Lattes CV parsed in the argument root contain
    published works that do not contain a DOI (Digital Object Identifier).
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, the Lattes CV declares that all cited works
        contain a DOI. If 1, there are works declared that do not
        have a DOI associated.
        printerror: if True, shows the errors in the command window.
    """
    DOIworksflag = 0

    x = root.findall('.//TRABALHOS-EM-EVENTOS')
    if not x:
        qtyworks = 0
    else:
        qtyworks = len(x[0].getchildren())

    if qtyworks > 0:
        for i in range(0, len(x[0])):
            try:
                dummy = x[0][i][0].attrib['DOI']
            except:
                dummy = ''
            if dummy == '':
                DOIworksflag = 1
                if printerror: print('Existe trabalho publicado sem DOI.')
                break
    return DOIworksflag

# Trabalhos: ver se tem página inicial

def test_work_page_start(root, printerror=False):
    """Verifies if the works declared in the Lattes CV parsed in the
    argument root contain a declared start page.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, the all works in the Lattes CV have a start
        page declared. If 1, at least one work hasn't a start
        page declared.
        printerror: if True, shows the errors in the command window.
    """
    workstartpageflag = 0

    x = root.findall('.//TRABALHOS-EM-EVENTOS')
    if not x:
        qtyworks = 0
    else:
        qtyworks = len(x[0].getchildren())

    if qtyworks > 0:
        for i in range(0, len(x[0])):
            try:
                dummy = x[0][i][1].attrib['PAGINA-INICIAL']
            except:
                dummy = ''
            if dummy == '':
                workstartpageflag = 1
                if printerror: print('Existe trabalho publicado sem' + \
                      ' identificação de página inicial.')
                break
    return workstartpageflag

# Trabalhos: ver se tem página final

def test_work_page_end(root, printerror=False):
    """Verifies if the works declared in the Lattes CV parsed in the
    argument root contain a declared end page.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, the all works in the Lattes CV have a end page
        declared. If 1, at least one work hasn't a end page declared.
        printerror: if True, shows the errors in the command window.
    """
    workendpageflag = 0

    x = root.findall('.//TRABALHOS-EM-EVENTOS')
    if not x:
        qtyworks = 0
    else:
        qtyworks = len(x[0].getchildren())

    if qtyworks > 0:
        for i in range(0, len(x[0])):
            
            try:
                dummy = x[0][i][1].attrib['PAGINA-FINAL']
            except:
                dummy = ''

            if dummy == '':
                workendpageflag = 1
                if printerror: print('Existe trabalho publicado sem' + \
                      ' identificação de página final.')
                break
    return workendpageflag

#Trabalhos: verificar nomes de autores:

def test_work_author_name(root, printerror=False):
    """Verifies if the names of works authors declared in the Lattes CV
    parsed in the argument root may contain abbreviations.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, the all work authors in the Lattes CV have their
        complete name. If 1, some author may have their name
        abbreviated.
        printerror: if True, shows the errors in the command window.
    """
    authornameflag = 0

    x = root.findall('.//TRABALHOS-EM-EVENTOS')
    if not x:
        qtyworks = 0
    else:
        qtyworks = len(x[0].getchildren())

    if qtyworks > 0:
        for i in range(0, len(x[0])):
            y = x[0][i].findall('AUTORES')
            for z in y:
                if '.' in z.attrib['NOME-COMPLETO-DO-AUTOR'] and \
                    authornameflag:
                    authornameflag = 1
                    if printerror: print('Possivel abreviação detectada ' + \
                          'em nome de autor de trabalho.')
                    break
    return authornameflag

#Trabalhos: verificar existencia de autores sem ID CNPQ:

def test_work_author_ID(root, printerror=False):
    """Verifies if the names of works authors declared in the Lattes CV
    parsed in the argument root contain a CNPQ ID associated.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, the all work authors in the Lattes CV have a
        CNPQ ID associated. If 1, some author doesn't have a CNPQ
        ID associated.
        printerror: if True, shows the errors in the command window.
    """
    authorIDflag = 0

    x = root.findall('.//TRABALHOS-EM-EVENTOS')
    if not x:
        qtyworks = 0
    else:
        qtyworks = len(x[0].getchildren())

    if qtyworks > 0:
        for i in range(0, len(x[0])):
            y = x[0][i].findall('AUTORES')
            for z in y:
                try:
                    dummy = z.attrib['NRO-ID-CNPQ']
                except:
                    dummy = ''
                if dummy == '' and authorIDflag:
                    authorIDflag = 1
                    if printerror: print('Autor de trabalho citado ' + \
                          'sem ID CNPq apresentado.')
                    break
    return authorIDflag

#consistencia do numero de artigos
def test_if_paper(root, printerror=False):
    """Verifies if the Lattes CV parsed in the argument root contains
    published papers.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, the Lattes CV declares that the resercher
        produced papers. If 1, the researcher hasn't declared that
        they published any papers.
        printerror: if True, shows the errors in the command window.
    """
    npapersflag = 0

    x = root.findall('.//ARTIGOS-PUBLICADOS')

    if len(x) > 0:
        npapers = len(x[0].getchildren())
    else:
        npapers = 0
        if printerror: print('Nenhum artigo publicado.')
        npapersflag = 1

    return npapersflag

#Consistencia da existencia de DOI em todos os artigos
def test_DOI_paper(root, printerror=False):
    """Verifies if the Lattes CV parsed in the argument root contain
    published papers that do not contain a DOI (Digital Object Identifier).
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, the Lattes CV declares that all cited papers
        contain a DOI. If 1, there are papers declared that do not
        have a DOI associated.
        printerror: if True, shows the errors in the command window.
    """
    DOIpapersflag = 0

    x = root.findall('.//ARTIGOS-PUBLICADOS')

    if len(x) > 0:
        npapers = len(x[0].getchildren())
    else:
        npapers = 0

    if npapers > 0:
        for i in range(0, len(x[0])):
            try:
                dummy = x[0][i][0].attrib['DOI']
            except:
                dummy = ''
            if dummy == '':
                DOIpapersflag = 1
                if printerror: print('Existe artigo publicado sem DOI.')
                break

    return DOIpapersflag

# Artigos em periódicos ou congressos: ver se tem página inicial
def test_paper_page_start(root, printerror=False):
    """Verifies if the papers declared in the Lattes CV parsed in the
    argument root contain a declared start page.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, then all papers in the Lattes CV have a start
        page declared. If 1, at least one papers hasn't a start
        page declared.
        printerror: if True, shows the errors in the command window.
    """
    paperstartpageflag = 0

    x = root.findall('.//ARTIGOS-PUBLICADOS')

    if len(x) > 0:
        npapers = len(x[0].getchildren())
    else:
        npapers = 0

    if npapers > 0:
        for i in range(0, len(x[0])):
            
            try:
                dummy = x[0][i][1].attrib['PAGINA-INICIAL']
            except:
                dummy = ''

            if dummy == '':
                paperstartpageflag = 1
                if printerror: print('Existe artigo publicado sem ' + \
                      'identificação de página inicial.')
                break

    return paperstartpageflag

# Artigos em periódicos ou congressos: ver se tem página final
def test_paper_page_end(root, printerror=False):
    """Verifies if the papers declared in the Lattes CV parsed in the
    argument root contain a declared end page.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, then all papers in the Lattes CV have a end page
        declared. If 1, at least one paper hasn't a end page declared.
        printerror: if True, shows the errors in the command window.
    """
    paperendpageflag = 0

    x = root.findall('.//ARTIGOS-PUBLICADOS')

    if len(x) > 0:
        npapers = len(x[0].getchildren())
    else:
        npapers = 0

    if npapers > 0:
        for i in range(0, len(x[0])):
            
            try:
                dummy = x[0][i][1].attrib['PAGINA-FINAL'] 
            except:
                dummy = ''
            if dummy == '':
                paperendpageflag = 1
                if printerror: print('Existe artigo publicado sem ' + \
                      'identificação de página final.')
                break
    return paperendpageflag

#Artigos: verificar nomes de autores:
def test_paper_author_name(root, printerror=False):
    """Verifies if the names of papers authors declared in the Lattes CV
    parsed in the argument root may contain abbreviations.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, the all paper authors in the Lattes CV have
        their complete name. If 1, some author may have their name
        abbreviated.
        printerror: if True, shows the errors in the command window.
    """
    authornameflag = 0

    x = root.findall('.//ARTIGOS-PUBLICADOS')

    if len(x) > 0:
        npapers = len(x[0].getchildren())
    else:
        npapers = 0

    if npapers > 0:
        for i in range(0, len(x[0])):
            y = x[0][i].findall('AUTORES')
            for z in y:
                if '.' in z.attrib['NOME-COMPLETO-DO-AUTOR'] and \
                    authornameflag:
                    authornameflag = 1
                    if printerror: print('Possivel abreviação detectada' + \
                          ' em nome de autor de artigo.')
                    break
    return authornameflag

#Artigos: verificar existencia de autores sem ID CNPQ:

def test_paper_author_ID(root, printerror=False):
    """Verifies if the names of papers authors declared in the Lattes CV
    parsed in the argument root contain a CNPQ ID associated.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, the all paper authors in the Lattes CV have a
        CNPQ ID associated. If 1, some author doesn't have a CNPQ
        ID associated.
        printerror: if True, shows the errors in the command window.
    """
    authorIDflag = 0

    x = root.findall('.//ARTIGOS-PUBLICADOS')

    if len(x) > 0:
        npapers = len(x[0].getchildren())
    else:
        npapers = 0

    if npapers > 0:
        for i in range(0, len(x[0])):
            y = x[0][i].findall('AUTORES')
            for z in y:
                try:
                    dummy = z.attrib['NRO-ID-CNPQ']
                except:
                    dummy = ''
                if dummy == '' and authorIDflag:
                    authorIDflag = 1
                    if printerror: print('Autor de artigo citado sem ' + \
                                         'ID CNPq apresentado.')
                    break

    return authorIDflag

# consistencia de Mestrado / Doutorado sem título ou orientador(es)
def test_adviser(root, printerror=False):
    """Verifies if the names of advisers of Masters Degrees and
    Doctorates are cited.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, then degrees have an adviser cited. If 1,
        some degree doesn't have an adviser associated.
        printerror: if True, shows the errors in the command window.
    """
    x = root.findall('.//FORMACAO-ACADEMICA-TITULACAO')
    adviserflag = 0

    if x != []:
        for i in range(0, len(x[0].getchildren())):
            if x[0][i].tag == "GRADUACAO":

                try:
                    dummy = x[0][i].attrib['NOME-DO-ORIENTADOR']
                except:
                    dummy = ""

                if dummy == "":
                    adviserflag = 1
                    if printerror: print('Nome do orientador de ' + \
                                         'graduação não fornecido.')

                try:
                    dummy = x[0][i].attrib['TITULO-DO-TRABALHO-DE-CONCLUSAO-DE-CURSO']
                except:
                    dummy = ""
                
                if dummy == "":
                    adviserflag = 1
                    if printerror: print('Título do trabalho de ' + \
                                         'conclusão de curso não fornecido.')

            elif x[0][i].tag == "MESTRADO":

                try:
                    dummy = x[0][i].attrib['NOME-COMPLETO-DO-ORIENTADOR']
                except:
                    dummy = ""
                
                if dummy== "":
                    adviserflag = 1
                    if printerror: print('Nome do orientador de ' + \
                                         'mestrado não fornecido.')

                try:
                    dummy = x[0][i].attrib['TITULO-DA-DISSERTACAO-TESE']
                except:
                    dummy = ""

                if dummy == "":
                    adviserflag = 1
                    if printerror: print('Título da dissertação de' + \
                                         ' mestrado não fornecido.')

            elif x[0][i].tag == "DOUTORADO":

                try:
                    dummy = x[0][i].attrib['NOME-COMPLETO-DO-ORIENTADOR']
                except:
                    dummy = ""

                if dummy == "":
                    adviserflag = 1
                    if printerror: print('Nome do orientador de ' + \
                                         'doutorado não fornecido.')

                try:
                    dummy = x[0][i].attrib['TITULO-DA-DISSERTACAO-TESE']
                except:
                    dummy = ""

                if dummy == "":
                    adviserflag = 1
                    if printerror: print('Título da tese de ' + \
                                         'doutorado não fornecido.')

    return adviserflag

# Não possui linhas de pesquisa
def test_research_line(root, printerror=False):
    """Verifies if the Lattes CV contains a line of research.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, then the Lattes CV contains a line of research.
        If 1, the Lattes CV doesn't contain a line of research.
        printerror: if True, shows the errors in the command window.
    """
    flagline = 0
    x = root.findall('.//*[@TITULO-DA-LINHA-DE-PESQUISA]')
    if x == []:
        flagline = 1
        if printerror: print('Linhas de pesquisa não fornecidas.')
    return flagline

# Não possui áreas de atuação
def test_areas(root, printerror=False):
    """Verifies if the Lattes CV contains actuation areas.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, then the Lattes CV contains actuation areas.
        If 1, the Lattes CV doesn't contain actuation areas.
        printerror: if True, shows the errors in the command window.
    """
    areasflag = 0
    x = root.findall('.//AREAS-DE-ATUACAO')

    if x == []:
        areasflag = 1
        if printerror: print('Áreas de atuação não fornecidas.')

    return areasflag

# Nomes de artigos duplicados
def test_paper_doubles(root, printerror=False):
    """Verifies if the Lattes CV contains papers with duplicate titles.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, then the Lattes CV contains papers with duplicate
        titles. If 1, the Lattes CV doesn't contain papers with
        duplicate titles.
        printerror: if True, shows the errors in the command window.
    """
    paperduplicatesflag = 0
    x = root.findall('.//*[@TITULO-DO-ARTIGO]')
    y = []

    for z in x:
        y.append(z.attrib['TITULO-DO-ARTIGO'])

    y = sorted(y)

    if len(set(y)) < len(y):
        if printerror: print('Nome de artigo duplicado encontrado.')
        paperduplicatesflag = 1
    return paperduplicatesflag

# Nomes de trabalhos duplicados
def test_work_doubles(root, printerror=False):
    """Verifies if the Lattes CV contains works with duplicate titles.
    Args:
        root: the Lattes CV that has been parsed with the ElementTree
        language.
        flag: if 0, then the Lattes CV contains works with duplicate
        titles. If 1, the Lattes CV doesn't contain works with
        duplicate titles.
        printerror: if True, shows the errors in the command window.
    """
    workduplicatesflag = 0
    x = root.findall('.//*[@TITULO-DO-TRABALHO]')
    y = []
    for z in x:
        y.append(z.attrib['TITULO-DO-TRABALHO'])
    y = sorted(y)
    if len(set(y)) < len(y):
        if printerror: print('Nome de trabalho duplicado encontrado.')
        workduplicatesflag = 1
    return workduplicatesflag

def get_test_frame(folderlist, printerror=False, savefile=True):
    """
    Generates a dataframe containing the results of the tests executed
    in the Lattes CVs of the parameter folder.
    Args:
        folder: a string containing the folder address that contains
        the Lattes CV files.
        printerror: if True, shows the errors in the command window.
        savefile: if True, saves the dataframe file in the working folder.
    """

    import pandas as pd
    import xml.etree.ElementTree as ET
    import zipfile
    from datetime import datetime
    import os
    import LattesLab as ll
    import LattesLab.consistency as con

    columns = ['Nome',
               'Graduation_Level',
               'Year_1st_grad',
               'Year_1st_master',
               'Year_1st_doc',
               'Year_1st_postdoc',
               'age_in_days',
               'summary_OK',
               'ID_OK',
               'e-mail_OK',
               'language_OK',
               'nation_OK',
               'Graduation_Concluded',
               'Masters_Concluded',
               'Doctorate_Concluded',
               'PostDoctorate_Concluded',
               'Degrees_Sequency_OK',
               'Works_are_cited',
               'Works_DOI_OK',
               'Works_page_start_OK',
               'Works_page_end_OK',
               'Works_Authors_names_OK',
               'Works_Authors_IDs_OK',
               'Papers_are_cited',
               'Papers_DOI_OK',
               'Papers_page_start_OK',
               'Papers_page_end_OK',
               'Papers_Authors_names_OK',
               'Papers_Authors_IDs_OK',
               'Advisers_OK',
               'Research_Line_OK',
               'Actuation_Area_OK',
               'NO_papers_duplicates',
               'NO_works_duplicates'
              ]

    lattesframe = pd.DataFrame(columns=columns)

    #verify if the parameter folderlist is a list

    if not isinstance(folderlist, list):
        folderlist = [folderlist]

    #filters the zip files

    ziplist = nonziplist = []
    [ziplist, nonziplist] = ll.get_files_list(folderlist)

    count = 0
    for rightname in ziplist:
        count += 1
        archive = zipfile.ZipFile(rightname, 'r')
        cvfile = archive.open(archive.namelist()[0], 'r')

        tree = ET.parse(cvfile)
        root = tree.getroot()

        nome = root[0].attrib['NOME-COMPLETO']

    for rightname in ziplist:
        count += 1
        archive = zipfile.ZipFile(rightname, 'r')
        cvfile = archive.open(archive.namelist()[0], 'r')

        tree = ET.parse(cvfile)
        root = tree.getroot()

        nome = root[0].attrib['NOME-COMPLETO']

        gradlevel = con.grad_level(root)
        
        [ano1grad, ngrad] = ll.get_grad_count(root, "GRADUACAO")
        [ano1master, nmaster] = ll.get_grad_count(root, "MESTRADO")
        [ano1phd, nphd] = ll.get_grad_count(root, "DOUTORADO")
        [ano1postdoc, nposdoc] = ll.get_grad_count(root, "POS-DOUTORADO")
 
        dayslate = con.test_update(root)       
        summaryflag = con.test_summary(root)
        idflag = con.test_id(root)
        emailflag = con.test_email(root)
        languageflag = con.test_language(root)
        nationflag = con.test_nationality(root)
        gradconclflag = con.test_grad_end(root, "GRADUACAO")
        masterconclflag = con.test_grad_end(root, "MESTRADO")
        docconclflag = con.test_grad_end(root, "DOUTORADO")
        posdocconclflag = con.test_grad_end(root, "POS-DOUTORADO")
        gradseqflag = con.test_grad_seq(root)
        nworksflag = con.test_if_work(root)
        DOIworksflag = con.test_DOI_work(root)
        workstartpageflag = con.test_work_page_start(root)
        workendpageflag = con.test_work_page_end(root)
        workauthornameflag = con.test_work_author_name(root)
        workauthorIDflag = con.test_work_author_ID(root)
        npapersflag = con.test_if_paper(root)
        DOIpapersflag = con.test_DOI_paper(root)
        paperstartpageflag = con.test_paper_page_start(root)
        paperendpageflag = con.test_paper_page_end(root)
        paperauthornameflag = con.test_paper_author_name(root)
        paperauthorIDflag = con.test_paper_author_ID(root)
        adviserflag = con.test_adviser(root)
        lineflag = con.test_research_line(root)
        areasflag = con.test_areas(root)
        papersduplicatesflag = con.test_paper_doubles(root)
        workduplicatesflag = con.test_work_doubles(root)



        x = [nome, gradlevel, ano1grad, ano1master, ano1phd,
             ano1postdoc, dayslate, summaryflag, idflag, 
             emailflag, languageflag,
             nationflag, gradconclflag, masterconclflag, docconclflag,
             posdocconclflag, gradseqflag, nworksflag, DOIworksflag,
             workstartpageflag, workendpageflag, workauthornameflag,
             workauthorIDflag, npapersflag, DOIpapersflag,
             paperstartpageflag, paperendpageflag, paperauthornameflag,
             paperauthorIDflag, adviserflag, lineflag, areasflag,
             papersduplicatesflag, workduplicatesflag]

        dummy = pd.DataFrame(data=[x], columns=columns)

        lattesframe = lattesframe.append(dummy)
        del dummy

    lattesframe = lattesframe.reset_index()
    lattesframe = lattesframe.drop('index', axis=1)

    if savefile:
        csvfile = "consistency_frame" + datetime.now().strftime('%Y%m%d%H%M%S') + ".csv"
        lattesframe.to_csv(os.path.join(os.getcwd(), csvfile), index=False)

    return lattesframe
