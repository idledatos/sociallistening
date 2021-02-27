import datetime
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
from dash.dependencies import Input, Output
import plotly.express as px
import tweepy
import dataset
from textblob import TextBlob
#from sqlalchemy.exc import ProgrammingError
import json
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
import time
from PIL import Image
import base64
from io import BytesIO
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate


### Tweepy credentials and setting up
consumer_key='mx2DBLn5UBlSAE72VqflYwchn'
consumer_secret='HZ99ImAXFVGJ6GaaOaH8EbYPJ7rEyqUKGx9obP9OFwI5dWAN9Q'
access_token='1188186968783556614-TYGRVuIEp5Vof7eLoucBSKqN6iljQe'
access_token_secret='Z5oeS0z7jjXKzf09p5bSOSKgNTdD0eMxxONvQXHjvlg9e'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


### Aux Functions
sid = SentimentIntensityAnalyzer()

def generate_wordcloud(data):
    wc = WordCloud(width=400, height=330, max_words=200,background_color='white',collocations = True).generate_from_frequencies(data)
    plt.figure(figsize=(14,12))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    return wc.to_image()

def clean(x):	
    doc = nlp(x)
    palabras = [(token.pos_,token.text) for token in doc]
    palabras_utiles = [x[1] for x in palabras if (x[0] in ['ADJ','NOUN','PROPN','VERB']) or (x[1]=='no')]
    return ' '.join(palabras_utiles)

def get_tweet_sentiment(tweet):
        analysis = TextBlob(tweet)
        polarity = sid.polarity_scores(str(analysis))
        return polarity['compound']

file2 = open(r"lexicons/negative_words_es.txt","r",encoding='utf-8')
file3 = open(r"lexicons/positive_words_es.txt","r",encoding='utf-8')
file2=file2.readlines()
file3=file3.readlines()

def social_listening_user(x):
    diccionario_tweets = {}
    for status in tweepy.Cursor(api.user_timeline, screen_name=x, tweet_mode="extended").items(100):
        try:
            texto = status.retweeted_status.full_text
            retweet = 'si'
        except:
            texto = status.full_text
            retweet = 'no'
        try:
            diccionario_tweets[status.created_at]=(texto,retweet,status.user.location,status.favorite_count,status.retweet_count)
        except:
            diccionario_tweets[status.created_at]=(texto,retweet,'No tiene',status.favorite_count,status.retweet_count)
    data = pd.DataFrame.from_dict(diccionario_tweets,orient = 'index',columns = ['Text','Is_retweet','Location','Favorite_count','Retweet_count'])
    
    data['mentions_rt'] = data.Text.apply(lambda x: re.findall(r'@\w+',x))
    mentions_rt_arr = []
    for ment in data.mentions_rt:
        mentions_rt_arr.extend(ment)
    try:
        top_mentions = [x for x in pd.array(mentions_rt_arr).value_counts().head(10).reset_index()['index']]
        top_mentions_2 = ', '.join([x+'\n' for x in top_mentions])
        print('''The most mentioned accounts are: {}'''.format(', '.join(top_mentions)))
    except:
        top_mentions_2 = 'There are no mentions'
        print('''There are no mentions''')
    
    
    data['Hashtags'] = data.Text.apply(lambda x: re.findall(r'#\w+',x))
    hashtag_arr = []
    for hashtag in data.Hashtags:
        hashtag_arr.extend(hashtag)
    try:
        top_hashtags = [x for x in pd.array(hashtag_arr).value_counts().head(10).reset_index()['index']]
        top_hashtags_2 = ', '.join([x+'\n' for x in top_hashtags])
        print('''The most used hashtags are: {}'''.format(', '.join(top_hashtags)))
    except:
        top_hashtags_2 = 'There are no hashtags'
        print('''There are no hashtags''')

    data['Text_2'] = data.Text.apply(lambda x: re.sub('<U\+[A-Z0-9]+>','', x))
    data['Text_2'] = data['Text_2'].apply(lambda x: re.sub('&amp;','&', x))
    data['Text_2'] = data['Text_2'].apply(lambda x: re.sub('(\r\n|&|amp|<|>|@|/)+','',x))
    data['Text_2'] = data['Text_2'].apply(lambda x: re.sub('https?://[A-Za-z0-9.-/]+','',x))
    data['Text_2'] = data['Text_2'].apply(lambda x: re.sub('t.co','',x))
    data['Text_2'] = data['Text_2'].apply(lambda x: re.sub('https','',x))
    
    
    
    most_liked = data.sort_values(by = 'Favorite_count',ascending = False)['Text_2'].reset_index()['Text_2'][0]
    most_retweet = data.sort_values(by = 'Retweet_count',ascending = False)['Text_2'].reset_index()['Text_2'][0]
    print('----------------------------------------------------------------------------------------------------------------')
    print('''The most liked tweet is: {}'''.format(most_liked))
    print('''The most retweeted tweet is: {}'''.format(most_retweet))
    print('----------------------------------------------------------------------------------------------------------------')
    
    language = detect(data['Text_2'].reset_index()['Text_2'][0])
    num_index = 0
    while (language!='en') & (language!='es'):
        num_index += 1
        language  = detect(data['Text_2'].reset_index()['Text_2'][num_index])
            
    if language == 'es':
        nlp = spacy.load('es_core_news_sm')
        neg_words=[x.strip('\n').lower() for x in file2]
        pos_words=[x.strip('\n').lower() for x in file3]
        data['Neg']=[sum([True for x in text.split() if x in neg_words]) for text in data.Text_2]
        data['Pos']=[sum([True for x in text.split() if x in pos_words]) for text in data.Text_2]

        data['Palabras']=[sum(True for x in text.split()) for text in data.Text_2]
        data['Ratio']=(data.Pos-data.Neg)/data.Palabras

        positive_tweets = [x for x in data.sort_values(by='Ratio',ascending=False).reset_index().iloc[:5,8]]
        pos_tweets_2 = ', '.join([x+'\n' for x in positive_tweets])
        negative_tweets = [x for x in data.sort_values(by='Ratio',ascending=True).reset_index().iloc[:5,8]]
        neg_tweets_2 = ', '.join([x+'\n' for x in negative_tweets])

        
        #print('''LEXICONS''')
        print('Some positive tweets: ')
        for x in positive_tweets:
            print(x)
        print('Some negative tweets: ')
        for x in negative_tweets:
            print(x)
        neg_perc = data[data['Ratio'] < 0].shape[0]/data.shape[0]
        pos_perc =data[data['Ratio'] > 0].shape[0]/data.shape[0]
        neu_perc = data[data['Ratio'] == 0].shape[0]/data.shape[0]
    
    elif language == 'en':
        nlp = spacy.load('en_core_web_sm')
        data['Compound'] = data.Text.apply(lambda x: get_tweet_sentiment(x))
        pos_tweets = [x for x in data.sort_values(by = 'Compound',ascending = False).reset_index().iloc[:5,1]]
        pos_tweets_2 = ', '.join([x+'\n' for x in pos_tweets])
        neg_tweets = [x for x in data.sort_values(by = 'Compound',ascending = True).reset_index().iloc[:5,1]]
        neg_tweets_2 = ', '.join([x+'\n' for x in neg_tweets])
        print('Some positive tweets: ')
        for x in pos_tweets:
            print(x)
        print('Some negative tweets: ')
        for x in neg_tweets:
            print(x)
        neg_perc = data[data['Compound'] < 0].shape[0]/data.shape[0]
        pos_perc = data[data['Compound'] > 0].shape[0]/data.shape[0]
        neu_perc = data[data['Compound'] == 0].shape[0]/data.shape[0]

    print('----------------------------------------------------------------------------------------------------------------')

    print('Percentage of positive tweets: {}'.format(pos_perc))
    print('Percentage of negative tweets: {}'.format(neg_perc))
    print('Percentage of neutral tweets: {}'.format(neu_perc))
    print('----------------------------------------------------------------------------------------------------------------')
    
    def clean(x):	
	    doc = nlp(x)
	    palabras = [(token.pos_,token.text) for token in doc]
	    palabras_utiles = [x[1] for x in palabras if (x[0] in ['ADJ','NOUN','PROPN','VERB']) or (x[1]=='no')]
	    return ' '.join(palabras_utiles)

    data['Clean_Text'] = data.Text_2.apply(lambda x: clean(x))
    if language == 'en':
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize 
        stop_words = set(stopwords.words('english'))
        def remove_stop(x):
            word_tokens = word_tokenize(x) 
            filtered_sentence = [w for w in word_tokens if w.lower() not in stop_words]
            return ' '.join(filtered_sentence)
        data['Clean_Text'] = data.Clean_Text.apply(lambda x: remove_stop(x))
        
    combined_tweets=' '.join(data.Clean_Text.values)
    combined_tweets = combined_tweets.split()
    counts = Counter(combined_tweets)
    
    print('Wordcloud for the last tweets')
    return generate_wordcloud(counts),top_mentions_2,top_hashtags_2,most_liked,most_retweet,float(neg_perc)*100,float(neu_perc)*100,float(pos_perc)*100,neg_tweets_2,pos_tweets_2
###Components

text_input = html.Div(
    	[dbc.Input(id="input", placeholder="Ingrese la cuenta a observar",type="text", debounce =True, n_submit = 0),
        	html.Br()])

toast_u = dbc.Toast(
    [html.P(id = 'mentions_top', className="mb-0")],style= {'margin-top':'25%'},
    header="Most mentioned users")

toast_h = dbc.Toast(
    [html.P(id = 'hashtags_top', className="mb-0")],style= {'margin-top':'25%'},
    header="Most used hashtags")

alert_1 =  dbc.Alert([html.P("Most liked Tweet",className="alert-heading"),
                   html.Hr(),
                   html.P(id = 'tweet_liked')], color="primary")

alert_2 = dbc.Alert([html.P("Most retweeted Tweet",className="alert-heading"),
                   html.Hr(),
                   html.P(id = 'tweet_retweet')], color="warning")

progress = dbc.Progress(
    [dbc.Progress('Negative',id = 'neg_value', color="danger", bar=True),
    dbc.Progress('Neutral',id = 'neu_value', bar=True),
    dbc.Progress('Positive',id = 'pos_value', color="success", bar=True)],
    multi=True)


cards_neg = dbc.Card([dbc.CardHeader('Some Negative Tweets'),dbc.CardBody([html.P(id = 'neg_tweets_text')])], color="danger", inverse=True)

cards_pos = dbc.Card([dbc.CardHeader('Some Positive Tweets'),dbc.CardBody([html.P(id = 'pos_tweets_text')])], color="success", inverse=True)

### Dash layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
app.layout = html.Div([
					dbc.Row([html.H2('Social Listening')],justify="center", align="center"),
					dbc.Row([
						dbc.Col(html.Div(),width = 2, align = 'center'),
						dbc.Col(text_input,width = 8),
						dbc.Col(html.Div(),width = 2)],justify = 'center'),
					dbc.Row([
                        dbc.Col(toast_u,width = {"size": 2,'offset' : 1}),
                        dbc.Col(html.Div(html.Img(id="image_wc"),style = {'height':'50%', 'width':'50%','margin-left':'20%'}),className="h-50",width ={"size": 6}),
                        dbc.Col(toast_h,width = {"size": 2})],no_gutters=True,),
                    dbc.Row([
                        dbc.Col(html.Div(),width = 1, align = 'center'),
                        dbc.Col(alert_1),
                        dbc.Col(alert_2),
                        dbc.Col(html.Div(),width = 1, align = 'center'),]),
                    dbc.Row([html.H6('Sentiment Analysis')],justify="center", align="center"),
                    dbc.Row([
                        dbc.Col(html.Div(),width = 2, align = 'center'),
                        dbc.Col(progress),
                        dbc.Col(html.Div(),width = 2, align = 'center')]),
                    dbc.Row([
                        dbc.Col(html.Div(),width = 1, align = 'center'),
                        dbc.Col(cards_neg),
                        dbc.Col(cards_pos),
                        dbc.Col(html.Div(),width = 1, align = 'center'),])])



### Callback
@app.callback([Output('image_wc', 'src'),
               Output('mentions_top','children'),
               Output('hashtags_top','children'),
               Output('tweet_liked','children'),
               Output('tweet_retweet','children'),
               Output('neg_value','value'),
               Output('neu_value','value'),
               Output('pos_value','value'),
               Output('neg_tweets_text','children'),
               Output('pos_tweets_text','children')],
              [Input('input','value'),
               Input('input','n_submit')])

### Callback function

def update_output(value,n_submit):
	changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
	if ("n_submit" in changed_id) & ((value is not None) or value!="" ):
		return social_listening_user(value)
	else:
		raise PreventUpdate
        



if __name__ == '__main__':
    app.run_server(debug=True)







