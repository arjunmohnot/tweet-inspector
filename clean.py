import pandas as pd
import re, string
import nltk, unicodedata
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

def splitclusters(s):
    """Generate the grapheme clusters for the string s. (Not the full
    Unicode text segmentation algorithm, but probably good enough for
    Devanagari.)

    """
    virama = u'\N{DEVANAGARI SIGN VIRAMA}'
    cluster = u''
    last = None
    for c in s:
        dog = unicodedata.category(c)
        if dog == 'Ll': continue
        cat = dog[0]
        if cat == 'M' or cat == 'L' and last == virama:
            cluster += c
        else:
            if cluster:
                yield cluster
            cluster = c
        last = c
    if cluster:
        yield cluster

def ppro_util(docs):

    stp = open('histpwords.txt', encoding = 'utf-8')
    for i in range(len(docs)):
        docs[i] = docs[i].lower()
        docs[i] = ' '.join([j for j in word_tokenize(docs[i])
                            if '#' not in j and '@' not in j])
        docs[i] = docs[i].translate(docs[i].maketrans('', '', '"'))
        docs[i] = re.sub(r'^https?:\/\/.*[\r\n]*', '', docs[i], flags=re.MULTILINE)
        docs[i] = docs[i].translate(docs[i].maketrans('', '', string.punctuation))
        docs[i] = ' '.join([re.sub("\d+", " ", j) for j in word_tokenize(docs[i]) if 'https' not in j and j not in stp])

    isvalid = lambda x: 48 <= ord(x) <= 57 or 97 <= ord(x) <= 122 or ord(x) == 32
    for i in range(len(docs)):
        docs[i] = ''.join(list(splitclusters(docs[i]))).strip()

    return docs


def clean(df):
#if __name__ == '__main__':
    #folder = 'C:/Users/gagan/Desktop'#/abuse/TweetScraper-master/TweetScraper/spiders/Data'
    #df = pd.read_excel(f'{folder}/data.xlsx', encoding='utf-8')
    print(df.shape)

    twval = df.value.tolist()
    twtxt = df.text.tolist()
    for i in range(len(twtxt)):
        twtxt[i] = ' '.join(str(twtxt[i]).split('\n'))
        twtxt[i] = ' '.join(str(twtxt[i]).split(' '))
    twtxt = ppro_util(twtxt)
    dfres = pd.DataFrame({'text': twtxt, 'value': twval})
    print(twtxt[:10])
    return dfres
    #dfres.to_excel('processed.xlsx', encoding='utf-8')

