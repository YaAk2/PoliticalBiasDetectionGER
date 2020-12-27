import numpy as np
import glob
from xml.etree import cElementTree as ET
import collections
#from newspaper import Article, Config
import pandas as pd
import matplotlib.pyplot as plt
import pickle
#import selenium
#from selenium import webdriver
import json

from itertools import chain
from collections import Counter
from operator import itemgetter
from more_itertools import unique_everseen

import nltk 
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

import string

def tokenize(texts):
    '''
    Split texts into tokens and get rid of punctuations and numbers
    '''
    if isinstance(texts, str):
        texts = [texts]
    texts_tokenized = [tokenizer.tokenize(t.lower()) for t in texts]
    return texts_tokenized
    
def remove_duplicates(texts, labels=None):
    if labels:
        data = zip(texts, labels)
        data_unique = unique_everseen(data, key=itemgetter(0))
        texts, labels = zip(*data_unique)
        return list(texts), list(labels)
    else:
        texts = list(dict.fromkeys(texts))
        return list(texts)

class GermaParl:
    def __init__(self, data_path):
        super().__init__()
        mapping = {'left': {'GRUENE', 'Bündnis 90/Die Grünen', 'PDS', 'DIE LINKE'}, 
                   'center': {'SPD', 'FDP'}, 
                   'right': {'CSU', 'CDU'}}
        self.texts = []
        self.labels = []
        parties = []
        for i in range(13, 19):
            xml_files = glob.glob(data_path + '/' + str(i) + '/*.xml')

            for f in xml_files:
                tree = ET.parse(f)
                root = tree.getroot()
                tex = root.findall('text')
                for te in tex:
                    body = te.findall('body')
                    for b in body:
                        div = b.findall('div')
                        for d in div:
                            sp = d.findall('sp')
                            for s in sp:
                                parties.append(s.attrib['party'])
                                p = s.findall('p')
                                if p:
                                    texts_per_speech = p[0].text
                                    for t in p[1:]:
                                        texts_per_speech = texts_per_speech + ' ' + t.text 
                                    self.texts.append(texts_per_speech)
        
        self.labels = ['left' if p in mapping['left']
                       else 'center' if p in mapping['center']
                       else 'right' for p in parties]
        
         
        self.texts, self.labels = remove_duplicates(self.texts, self.labels)    
        
        self.texts = tokenize(self.texts)
        
        self.number_of_samples = len(self.texts)
        

class Archive:
    def __init__(self, data_path):
        super().__init__()
        self.mapping = {'left': {'jungleworld_archive.pkl'}, 
                        'center': {'sueddeutsche_archive.pkl'}, 
                        'right': {'jungefreiheit_archive.pkl'}}

        with open(data_path, 'rb') as f:
            self.texts = pickle.load(f)
        
        self.texts = remove_duplicates(self.texts)
        
        if data_path.split('/')[1] =='jungefreiheit_archive.pkl':
            tokenized_jf = [tokenizer.tokenize(t.lower()) for t in self.texts]
            ###Drop link and date of article, and css formats
            tokenized_jf = tokenized_jf[:15243] + [t[13:] for t in tokenized_jf[15243:56207]] + \
            [t[97:] for t in tokenized_jf[56207:58558]] + [t[102:] for t in tokenized_jf[58558:70687]] + \
            [t[105:] for t in tokenized_jf[70687:]]
            self.texts = tokenized_jf
        else:
            self.texts = tokenize(self.texts)
        
        self.number_of_samples = len(self.texts)
    
        self.labels = []
        if data_path.split('/')[1] in self.mapping['left']:
            self.labels.extend(['left']*len(self.texts))
        elif data_path.split('/')[1] in self.mapping['right']:
            self.labels.extend(['right']*len(self.texts))
        elif data_path.split('/')[1] in self.mapping['center']:
            self.labels.extend(['center']*len(self.texts))
        
class ScrapeArchive:
    def __init__(self):
        super().__init__()
        self.driver = webdriver.Chrome('chromedriver.exe')
        self.config = Config()
        self.config.memoize_articles = False
        self.config.request_timeout = 10
        
        
    def jf(self):
        '''Junge Freiheit Archive'''
        years = ['', '98', '99', '00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
                 '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
        jungefreiheit_texts = []

        for y in years: 
            while True:
                try:
                    self.driver.get('https://jf-archiv.de/archiv' + y + '.htm')
                except selenium.common.exceptions.WebDriverException:
                    continue
                    
            elems = self.driver.find_elements_by_xpath("//a[@href]")
            urls_to_visit = []

            for elem in elems[1:]:
                urls_to_visit.append(elem.get_attribute('href'))

            for url in urls_to_visit:
                while True:
                    try:
                        self.driver.get(url)
                        break
                    except selenium.common.exceptions.WebDriverException:
                        continue
                el = self.driver.find_elements_by_xpath("//a[@href]")
                for e in el[1:-4]:
                    u = str(e.get_attribute('href'))
                    article = Article(u, config=self.config)
                    article.download()
                    try:
                        article.parse()
                        jungefreiheit_texts.append(article.text)
                    except newspaper.article.ArticleException:
                        continue
        return jungefreiheit_texts


    def jw(self):
        '''Junngle World Archive'''
        years = ['1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', 
                 '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
        jungleworld_texts = []
                
        for y in years:
            for w in range(1, 53): #weeks
                url = 'https://jungle.world/inhalt/' + str(y) + '/' + f'{w:02d}'
                while True:
                    try:
                        self.driver.get(url)
                        break
                    except selenium.common.exceptions.WebDriverException:
                        continue
                
                el = self.driver.find_elements_by_xpath('//a[contains(@href,"/artikel/")]')
                
                for e in el:
                    u = str(e.get_attribute('href'))
                    article = Article(u, config=self.config)
                    article.download()
                    try:
                        article.parse()
                        jungleworld_texts.append(article.text)
                    except newspaper.article.ArticleException:
                        continue
        return jungleworld_texts


    def sz(self):
        '''Sueddeutsche Zeitung Archive'''
        themes = ['politik', 'wirtschaft', 'panorama', 'sport', 'kultur', 'wissen', 'leben']
        years = ['2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', 
                 '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        sueddeutsche_texts = []
        
        for t in themes:
            for y in years:
                for m in range(1, 13): #months
                    url = 'https://www.sueddeutsche.de/archiv/'+ t + '/' + y + '/' + f'{m:02d}'
                    while True:
                        try:
                            self.driver.get(url)
                            break
                        except selenium.common.exceptions.WebDriverException:
                            continue           
                    
                    xpath = '//a[contains(@href,' + '"/archiv/' + t + '/' + y + '/' + str(m) + '/page/"' + ')]'
                    pages = self.driver.find_elements_by_xpath(xpath)
                    
                    try:
                        num_pages = max([int((p.get_attribute('href')).split("/")[-1]) for p in pages])
                    except ValueError:
                        num_pages = 1
                    
                    for p in range(1, num_pages + 1):
                        url = 'https://www.sueddeutsche.de/archiv/'+ t + '/' + y + '/' + f'{m:02d}' + '/page/' + str(p)
                        while True:
                            try:
                                self.driver.get(url)
                                break
                            except selenium.common.exceptions.WebDriverException:
                                continue                          
                        
                        elems = self.driver.find_elements_by_xpath('//a[contains(@href,"https://www.sueddeutsche.de/politik/")]')   
                        
                        for elem in elems:
                            u = str(elem.get_attribute('href'))
                            article = Article(u, config=self.config, language='de')
                            article.download()
                            try:
                                article.parse()
                                sueddeutsche_texts.append(article.text)
                            except newspaper.article.ArticleException:
                                continue
        
        return sueddeutsche_texts 
            
            
class ScrapeLatestNews:
    def __init__(self):
        
        self.texts = []
        
        '''Preußische Allgemeine Zeitung'''
        paz_politics = newspaper.build('https://paz.de/rubrik/politik', memoize_articles=False)
        paz_culture = newspaper.build('https://paz.de/rubrik/kultur', memoize_articles=False)
        paz_economics = newspaper.build('https://paz.de/rubrik/wirtschaft', memoize_articles=False)
        paz_society = newspaper.build('https://paz.de/rubrik/gesellschaft', memoize_articles=False)
        paz_history = newspaper.build('https://paz.de/rubrik/geschichte', memoize_articles=False)

        paz_articles = paz_politics.articles + paz_culture.articles + paz_economics.articles + \
        paz_society.articles + paz_history.articles
        
        for i in range(len(paz_articles)):
            a = Article(paz_articles[i].url)
            a.download()
            a.parse()
            self.texts.append(a.text)
          
        self.labels = ['right']*len(paz_articles)
        
        
        '''Junge Freiheit'''
        jungefreiheit_economics = newspaper.build('https://jungefreiheit.de/wirtschaft/', memoize_articles=False)
        jungefreiheit_politics = newspaper.build('https://jungefreiheit.de/politik/', memoize_articles=False)
        jungefreiheit_culture = newspaper.build('https://jungefreiheit.de/kultur/', memoize_articles=False)
        jungefreiheit_debate = newspaper.build('https://jungefreiheit.de/debatte-2/', memoize_articles=False)
        jungefreiheit_knowledge = newspaper.build('https://jungefreiheit.de/wissen/', memoize_articles=False)

        jungefreiheit_articles = jungefreiheit_economics.articles + jungefreiheit_politics.articles + jungefreiheit_culture.articles + \
        jungefreiheit_debate.articles + jungefreiheit_knowledge.articles
        
        for i in range(len(jungefreiheit_articles)):
            a = Article(jungefreiheit_articles[i].url)
            a.download()
            a.parse()
            self.texts.append(a.text)
          
        self.labels.extend(['right']*len(jungefreiheit_articles))
        
        
        '''Die Zeit'''
        diezeit_politics = newspaper.build('https://www.zeit.de/politik/index', memoize_articles=False)
        diezeit_society = newspaper.build('https://www.zeit.de/gesellschaft/index', memoize_articles=False)
        diezeit_economics = newspaper.build('https://www.zeit.de/wirtschaft/index', memoize_articles=False)
        diezeit_culture = newspaper.build('https://www.zeit.de/kultur/index', memoize_articles=False)
        diezeit_knowledge = newspaper.build('https://www.zeit.de/wissen/index', memoize_articles=False)

        diezeit_articles = diezeit_politics.articles + diezeit_society.articles + diezeit_economics.articles + \
        diezeit_culture.articles + diezeit_knowledge.articles
        
        for i in range(len(diezeit_articles)):
            a = Article(diezeit_articles[i].url)
            a.download()
            a.parse()
            self.texts.append(a.text)
          
        self.labels.extend(['center']*len(diezeit_articles))
        
  
        '''tagesspiegel'''
        tagesspiegel_politics = newspaper.build('https://www.tagesspiegel.de/politik/', memoize_articles=False)
        tagesspiegel_economics = newspaper.build('https://www.tagesspiegel.de/wirtschaft/', memoize_articles=False)
        tagesspiegel_society = newspaper.build('https://www.tagesspiegel.de/gesellschaft/', memoize_articles=False)
        tagesspiegel_culture = newspaper.build('https://www.tagesspiegel.de/kultur/', memoize_articles=False)
        tagesspiegel_knowledge = newspaper.build('https://www.tagesspiegel.de/wissen/', memoize_articles=False)
        tagesspiegel_debate = newspaper.build('https://www.tagesspiegel.de/meinung/', memoize_articles=False)

        tagesspiegel_articles = tagesspiegel_politics.articles + tagesspiegel_society.articles + tagesspiegel_economics.articles + \
        tagesspiegel_culture.articles + tagesspiegel_knowledge.articles + tagesspiegel_debate.articles
        
        for i in range(len(tagesspiegel_articles)):
            a = Article(tagesspiegel_articles[i].url)
            a.download()
            a.parse()
            self.texts.append(a.text)
          
        self.labels.extend(['center']*len(tagesspiegel_articles))
       
        
        
        '''handelsblatt'''
        handelsblatt_politics = newspaper.build('https://www.handelsblatt.com/politik/?navi=POLITIK_1979524', memoize_articles=False)
        handelsblatt_industry = newspaper.build('https://www.handelsblatt.com/unternehmen/?navi=UNTERNEHMEN_1980072',
                                                memoize_articles=False)
        handelsblatt_finance = newspaper.build('https://www.handelsblatt.com/finanzen/?navi=FINANZEN_1980476', 
                                               memoize_articles=False)
        handelsblatt_debate = newspaper.build('https://www.handelsblatt.com/meinung/?navi=MEINUNG_4671738', memoize_articles=False)

        handelsblatt_articles = handelsblatt_politics.articles + handelsblatt_industry.articles + \
        handelsblatt_finance.articles + handelsblatt_debate.articles
        
        for i in range(len(handelsblatt_articles)):
            a = Article(handelsblatt_articles[i].url)
            a.download()
            a.parse()
            self.texts.append(a.text)
          
        self.labels.extend(['center']*len(handelsblatt_articles))
        
        
        '''welt'''
        welt_politics = newspaper.build('https://www.welt.de/politik/', memoize_articles=False)
        welt_economics = newspaper.build('https://www.welt.de/wirtschaft/', memoize_articles=False)
        welt_mix = newspaper.build('https://www.welt.de/vermischtes/', memoize_articles=False)
        welt_knowledge = newspaper.build('https://www.welt.de/wissenschaft/', memoize_articles=False)
        welt_culture = newspaper.build('https://www.welt.de/kultur/', memoize_articles=False)
        welt_history = newspaper.build('https://www.welt.de/geschichte/', memoize_articles=False)
        
        welt_articles = welt_politics.articles + welt_economics.articles + welt_mix.articles + welt_knowledge.articles + \
        welt_culture.articles + welt_history.articles
        
        for i in range(len(welt_articles)):
            a = Article(welt_articles[i].url)
            a.download()
            a.parse()
            self.texts.append(a.text)
          
        self.labels.extend(['center']*len(welt_articles))
        
    
        '''faz'''
        faz_politics = newspaper.build('https://www.faz.net/aktuell/politik/', memoize_articles=False)
        faz_economics = newspaper.build('https://www.faz.net/aktuell/wirtschaft/', memoize_articles=False)
        faz_finance = newspaper.build('https://www.faz.net/aktuell/finanzen/', memoize_articles=False)
        faz_feuilleton = newspaper.build('https://www.faz.net/aktuell/feuilleton/', memoize_articles=False)
        faz_society = newspaper.build('https://www.faz.net/aktuell/gesellschaft/', memoize_articles=False)
        faz_knowledge = newspaper.build('https://www.faz.net/aktuell/wissen/', memoize_articles=False)
        
        faz_articles = faz_politics.articles + faz_economics.articles + faz_finance.articles + faz_feuilleton.articles + \
        faz_society.articles + faz_knowledge.articles
        
        for i in range(len(faz_articles)):
            a = Article(faz_articles[i].url)
            a.download()
            a.parse()
            self.texts.append(a.text)
          
        self.labels.extend(['center']*len(faz_articles))
        
      
        '''sz'''
        sz_politics = newspaper.build('https://www.sueddeutsche.de/politik', memoize_articles=False)
        sz_economics = newspaper.build('https://www.sueddeutsche.de/wirtschaft', memoize_articles=False)
        sz_debate = newspaper.build('https://www.sueddeutsche.de/meinung', memoize_articles=False)
        sz_mix = newspaper.build('https://www.sueddeutsche.de/panorama', memoize_articles=False)
        sz_culture = newspaper.build('https://www.sueddeutsche.de/kultur', memoize_articles=False)
        sz_society = newspaper.build('https://www.sueddeutsche.de/leben', memoize_articles=False)
        sz_knowledge = newspaper.build('https://www.sueddeutsche.de/wissen', memoize_articles=False)
        
        sz_articles = sz_politics.articles + sz_economics.articles + sz_debate.articles + sz_mix.articles + sz_culture.articles + \
        sz_society.articles + sz_knowledge.articles
        
        for i in range(len(faz_articles)):
            a = Article(faz_articles[i].url)
            a.download()
            a.parse()
            self.texts.append(a.text)
          
        self.labels.extend(['center']*len(faz_articles))
        
        
        '''Jungle World'''
        jungleworld = newspaper.build('https://jungle.world/', memoize_articles=False)
        jungleworld_articles = jungleworld.articles
        
        for i in range(len(jungleworld_articles)):
            a = Article(jungleworld_articles[i].url)
            a.download()
            a.parse()
            self.texts.append(a.text)
            
        self.labels.extend(['left']*len(jungleworld_articles))
        
        
        '''Der Freitag'''
        derfreitag_politics = newspaper.build('https://www.freitag.de/politik', memoize_articles=False)
        derfreitag_economics = newspaper.build('https://www.freitag.de/wirtschaft', memoize_articles=False)
        derfreitag_culture = newspaper.build('https://www.freitag.de/kultur', memoize_articles=False)
        derfreitag_debate = newspaper.build('https://www.freitag.de/debatte', memoize_articles=False)
        
        derfreitag_articles = derfreitag_politics.articles + derfreitag_economics.articles + \
        derfreitag_culture.articles + derfreitag_debate.articles
        
        for i in range(len(derfreitag_articles)):
            a = Article(derfreitag_articles[i].url)
            a.download()
            a.parse()
            self.texts.append(a.text)
            
        self.labels.extend(['left']*len(derfreitag_articles))

        '''Junge Welt'''
        jungewelt_interior = newspaper.build('https://www.jungewelt.de/aktuell/rubrik/inland.php', memoize_articles=False)
        jungewelt_foreign = newspaper.build('https://www.jungewelt.de/aktuell/rubrik/ausland.php', memoize_articles=False)
        jungewelt_labor = newspaper.build('https://www.jungewelt.de/aktuell/rubrik/kapital_und_arbeit.php', memoize_articles=False)
        jungewelt_feuilleton = newspaper.build('https://www.jungewelt.de/aktuell/rubrik/feuilleton.php', memoize_articles=False)
        jungewelt_themes = newspaper.build('https://www.jungewelt.de/aktuell/rubrik/thema.php', memoize_articles=False)
        
        jungewelt_articles = jungewelt_interior.articles + jungewelt_foreign.articles + jungewelt_labor.articles + \
        jungewelt_feuilleton.articles + jungewelt_themes.articles
        
        for i in range(len(jungewelt_articles)):
            a = Article(jungewelt_articles[i].url)
            a.download()
            a.parse()
            self.texts.append(a.text)
            
        self.labels.extend(['left']*len(jungewelt_articles))
        
        '''Neues Deutschland'''
        neuesdeutschland_politics = newspaper.build('https://www.neues-deutschland.de/rubrik/politik/', memoize_articles=False)
        neuesdeutschland_economics = newspaper.build('https://www.neues-deutschland.de/rubrik/wirtschaft-umwelt/', memoize_articles=False)
        neuesdeutschland_movement = newspaper.build('https://www.neues-deutschland.de/rubrik/bewegung/', memoize_articles=False)
        neuesdeutschland_debate = newspaper.build('https://www.neues-deutschland.de/nd-debatte/', memoize_articles=False)
        neuesdeutschland_culture = newspaper.build('https://www.neues-deutschland.de/rubrik/kultur/', memoize_articles=False)
        neuesdeutschland_knowledge = newspaper.build('https://www.neues-deutschland.de/rubrik/wissen/', memoize_articles=False)
        
        neuesdeutschland_articles = neuesdeutschland_politics.articles + neuesdeutschland_economics.articles + \
        neuesdeutschland_movement.articles + neuesdeutschland_debate.articles + neuesdeutschland_culture.articles + \
        neuesdeutschland_knowledge
        
        for i in range(len(neuesdeutschland_articles)):
            a = Article(neuesdeutschland_articles[i].url)
            a.download()
            a.parse()
            self.texts.append(a.text)
            
        self.labels.extend(['left']*len(neuesdeutschland_articles))
        
     
        '''taz'''
        taz_politics = newspaper.build('https://taz.de/Politik/!p4615/', memoize_articles=False)
        taz_economics = newspaper.build('https://taz.de/Oeko/!p4610/', memoize_articles=False)
        taz_society = newspaper.build('https://taz.de/Gesellschaft/!p4611/', memoize_articles=False)
        taz_culture = newspaper.build('https://taz.de/Kultur/!p4639/', memoize_articles=False)
        
        taz_articles = taz_politics.articles + taz_economics.articles + taz_society.articles + taz_culture.articles
        
        for i in range(len(taz_articles)):
            a = Article(taz_articles[i].url)
            a.download()
            a.parse()
            self.texts.append(a.text)
            
        self.labels.extend(['left']*len(taz_articles))        
        
        self.number_of_samples = len(self.texts)

        
def data_preprocessing(texts, drop_stopwords, drop_digits):
    '''
    drop_stopwords: Boolean, if True stopwords are dropped.
    drop_digits: Boolean, if True digits are dropped.
    '''
    
    stopwords = []
    if drop_stopwords:
        file = open("stopwords-ger/stopwords-ger.txt")
        file_contents = file.read()
        stopwords = file_contents.splitlines()
    
    texts_prepocessed = []
    for t in texts:
        texts_prepocessed.append([w for w in t if w not in stopwords and w.isdigit() is not drop_digits and len(w) !=1])
        
    return texts_prepocessed
        
def plot_num_samples_per_class(labels):
    df = pd.DataFrame({'label': labels})
    df.groupby('label', as_index=True).size().plot(kind='bar')
    plt.title('Number of samples per class')
    plt.show()


def num_words_per_sample(texts):
    return np.median([len(s) for s in texts])


def plot_sample_length_distribution(texts):
    plt.hist([len(s) for s in texts], bins=80)
    plt.xlim(0, max([len(s) for s in texts]))
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()

    
def plot_word_frequency_distribution(texts):
    top_k = 40
    w = Counter(chain.from_iterable(texts))
    word, frequency = zip(*w.most_common()[:top_k])
    indices = np.arange(len(w.most_common()[:top_k]))
    plt.bar(indices, frequency)
    plt.xticks(indices, word, rotation='vertical')
    plt.xlabel('Words')
    plt.ylabel('Frequencies')
    plt.title('Frequency distribution of words')
    plt.show()
    return Counter(chain.from_iterable(texts))
    
