import pandas as pd
import random

folder = 'C:/Users/gagan/Desktop/abuse/TweetScraper-master/TweetScraper/spiders/Data'
adf = pd.read_excel(f'{folder}/tweets3.xlsx', encoding = 'utf-8')
nadf = pd.read_excel(f'{folder}/tweets4.xlsx', encoding = 'utf-8')
#adf = adf.head(1111)
adf.insert(adf.shape[1]-1, 'value', 1)
nadf.insert(nadf.shape[1]-1, 'value', 0)

adf = adf.append(nadf.append(nadf))

adf.drop('has_image', axis=1, inplace=True)
adf.drop('has_video', axis=1, inplace=True)
adf.drop('has_media', axis=1, inplace=True)
adf.drop('images', axis=1, inplace=True)
adf.drop('videos', axis=1, inplace=True)
adf.drop('medias', axis=1, inplace=True)
adf.drop('Unnamed: 0', axis=1, inplace=True)
adf = adf.sample(frac=1)
adf.to_excel('data.xlsx', encoding='utf-8')
