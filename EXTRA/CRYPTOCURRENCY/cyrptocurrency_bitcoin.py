
# coding: utf-8

# In[1]:

import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np


# In[2]:

bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]


# In[3]:

bitcoin_market_info.head()


# In[4]:

bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))


# In[5]:

bitcoin_market_info.loc[bitcoin_market_info['Volume']=="-",'Volume']=0


# In[6]:

bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype('int64')


# In[7]:

bitcoin_market_info.head()


# In[8]:

eth_market_info = pd.read_html("https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]


# In[9]:

eth_market_info = eth_market_info.assign(Date=pd.to_datetime(eth_market_info['Date']))


# In[10]:

eth_market_info.head()


# In[11]:

ripple_market_info = pd.read_html("https://coinmarketcap.com/currencies/ripple/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
ripple_market_info = ripple_market_info.assign(Date=pd.to_datetime(ripple_market_info['Date']))
ripple_market_info.loc[ripple_market_info['Volume']=="-",'Volume']=0
ripple_market_info['Volume'] = ripple_market_info['Volume'].astype('int64')
ripple_market_info.head()


# In[12]:

import sys
from PIL import Image
import io

if sys.version_info[0] < 3:
    import urllib2 as urllib
    bt_img = urllib.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")
    eth_img = urllib.urlopen("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ethereum_logo_2014.svg/256px-Ethereum_logo_2014.svg.png")
    ripple_img = urllib.urlopen("data:image/png;base64,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")
else:
    import urllib
    bt_img = urllib.request.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")
    eth_img = urllib.request.urlopen("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ethereum_logo_2014.svg/256px-Ethereum_logo_2014.svg.png")
    ripple_img = urllib.request.urlopen("data:image/png;base64,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")


image_file = io.BytesIO(bt_img.read())
bitcoin_im = Image.open(image_file)

image_file = io.BytesIO(eth_img.read())
eth_im = Image.open(image_file)
width_eth_im , height_eth_im  = eth_im.size
eth_im = eth_im.resize((int(eth_im.size[0]*0.8), int(eth_im.size[1]*0.8)), Image.ANTIALIAS)

image_file = io.BytesIO(ripple_img.read())
ripple_im = Image.open(image_file)


# In[13]:

bitcoin_market_info.columns =[bitcoin_market_info.columns[0]]+['bt_'+i for i in bitcoin_market_info.columns[1:]]
eth_market_info.columns =[eth_market_info.columns[0]]+['eth_'+i for i in eth_market_info.columns[1:]]
ripple_market_info.columns =[ripple_market_info.columns[0]]+['ripple_'+i for i in ripple_market_info.columns[1:]]


# In[14]:

fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})
ax1.set_ylabel('Closing Price ($)',fontsize=12)
ax2.set_ylabel('Volume ($ bn)',fontsize=12)
ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
ax2.set_yticklabels(range(10))
ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
ax1.plot(bitcoin_market_info['Date'].astype(datetime.datetime),bitcoin_market_info['bt_Open'])
ax2.bar(bitcoin_market_info['Date'].astype(datetime.datetime).values, bitcoin_market_info['bt_Volume'].values)
fig.tight_layout()
fig.figimage(bitcoin_im, 50, 20, zorder=3,alpha=.5)
plt.show()


# In[15]:

fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})
#ax1.set_yscale('log')
ax1.set_ylabel('Closing Price ($)',fontsize=12)
ax2.set_ylabel('Volume ($ bn)',fontsize=12)
ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
ax2.set_yticklabels(range(10))
ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
ax1.plot(eth_market_info['Date'].astype(datetime.datetime),eth_market_info['eth_Open'])
ax2.bar(eth_market_info['Date'].astype(datetime.datetime).values, eth_market_info['eth_Volume'].values)
fig.tight_layout()
fig.figimage(eth_im, 300, 180, zorder=3, alpha=.6)
plt.show()


# In[16]:

fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})
ax1.set_ylabel('Closing Price ($)',fontsize=12)
ax2.set_ylabel('Volume ($ bn)',fontsize=12)
ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
ax2.set_yticklabels(range(10))
ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
ax1.plot(ripple_market_info['Date'].astype(datetime.datetime),ripple_market_info['ripple_Open'])
ax2.bar(ripple_market_info['Date'].astype(datetime.datetime).values, ripple_market_info['ripple_Volume'].values)
fig.tight_layout()
fig.figimage(ripple_im, 50, 20, zorder=3,alpha=.5)
plt.show()


# In[17]:

market_info = pd.merge(bitcoin_market_info,eth_market_info, on=['Date'])
market_info = pd.merge(market_info,ripple_market_info, on=['Date'])


# In[18]:

market_info.head()


# In[19]:

market_info = market_info[market_info['Date']>='2016-01-01']
for coins in ['bt_', 'eth_','ripple_']: 
    kwargs = { coins+'day_diff': lambda x: (x[coins+'Close']-x[coins+'Open'])/x[coins+'Open']}
    market_info = market_info.assign(**kwargs)
market_info.head()


# In[29]:

split_date = '2017-06-01'
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])


# In[32]:

ax3.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
ax3.set_xticklabels('')


# In[33]:

ax1.plot(market_info[market_info['Date'] < split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] < split_date]['bt_Close'], 
         color='#B08FC7', label='Training')
ax1.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] >= split_date]['bt_Close'], 
         color='#8FBAC8', label='Test')
ax2.plot(market_info[market_info['Date'] < split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] < split_date]['eth_Close'], 
         color='#B08FC7')
ax2.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] >= split_date]['eth_Close'], color='#8FBAC8')
                    
ax3.plot(market_info[market_info['Date'] < split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] < split_date]['ripple_Close'], 
         color='#B08FC7')
ax3.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] >= split_date]['ripple_Close'], color='#8FBAC8')


# In[34]:

ax1.set_xticklabels('')
ax3.set_xticklabels('')
ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
ax2.set_ylabel('Ethereum Price ($)',fontsize=12)
ax3.set_ylabel('Ripple Price ($)',fontsize=12)


# In[37]:

plt.tight_layout()
ax1.legend(bbox_to_anchor=(0.03, 1), loc=2, borderaxespad=0., prop={'size': 14})
fig.figimage(bitcoin_im.resize((int(bitcoin_im.size[0]*0.65), int(bitcoin_im.size[1]*0.65)), Image.ANTIALIAS), 
             200, 260, zorder=3,alpha=.5)
fig.figimage(eth_im.resize((int(eth_im.size[0]*0.65), int(eth_im.size[1]*0.65)), Image.ANTIALIAS), 
             350, 40, zorder=3,alpha=.5)
fig.figimage(ripple_im.resize((int(ripple_im.size[0]*0.65), int(ripple_im.size[1]*0.65)), Image.ANTIALIAS), 
             350, 40, zorder=3,alpha=.5)


# In[38]:

plt.show()


# In[39]:

# trivial lag model: P_t = P_(t-1)
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax2.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y')  for i in range(12)])
ax3.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax3.set_xticklabels('')
ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date']>= split_date]['bt_Close'].values, label='Actual')
ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
          market_info[market_info['Date']>= datetime.datetime.strptime(split_date, '%Y-%m-%d') - 
                      datetime.timedelta(days=1)]['bt_Close'][1:].values, label='Predicted')
ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
ax1.set_title('Simple Lag Model (Test Set)')
ax2.set_ylabel('Etherum Price ($)',fontsize=12)
ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date']>= split_date]['eth_Close'].values, label='Actual')
ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
          market_info[market_info['Date']>= datetime.datetime.strptime(split_date, '%Y-%m-%d') - 
                      datetime.timedelta(days=1)]['eth_Close'][1:].values, label='Predicted')

ax3.set_ylabel('Ripple Price ($)',fontsize=12)
ax3.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date']>= split_date]['ripple_Close'].values, label='Actual')
ax3.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
          market_info[market_info['Date']>= datetime.datetime.strptime(split_date, '%Y-%m-%d') - 
                      datetime.timedelta(days=1)]['ripple_Close'][1:].values, label='Predicted')
fig.tight_layout()
plt.show()



# In[40]:

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.hist(market_info[market_info['Date']< split_date]['bt_day_diff'].values, bins=100)
ax2.hist(market_info[market_info['Date']< split_date]['eth_day_diff'].values, bins=100)
ax3.hist(market_info[market_info['Date']< split_date]['ripple_day_diff'].values, bins=100)
ax1.set_title('Bitcoin Daily Price Changes')
ax2.set_title('Ethereum Daily Price Changes')
ax3.set_title('Ripple Daily Price Changes')
plt.show()


# In[41]:

np.random.seed(202)
bt_r_walk_mean, bt_r_walk_sd = np.mean(market_info[market_info['Date']< split_date]['bt_day_diff'].values),                          np.std(market_info[market_info['Date']< split_date]['bt_day_diff'].values)
bt_random_steps = np.random.normal(bt_r_walk_mean, bt_r_walk_sd, 
                (max(market_info['Date']).to_pydatetime() - datetime.datetime.strptime(split_date, '%Y-%m-%d')).days + 1)
eth_r_walk_mean, eth_r_walk_sd = np.mean(market_info[market_info['Date']< split_date]['eth_day_diff'].values),                          np.std(market_info[market_info['Date']< split_date]['eth_day_diff'].values)
eth_random_steps = np.random.normal(eth_r_walk_mean, eth_r_walk_sd, 
                (max(market_info['Date']).to_pydatetime() - datetime.datetime.strptime(split_date, '%Y-%m-%d')).days + 1)

ripple_r_walk_mean, ripple_r_walk_sd = np.mean(market_info[market_info['Date']< split_date]['ripple_day_diff'].values),                          np.std(market_info[market_info['Date']< split_date]['ripple_day_diff'].values)
ripple_random_steps = np.random.normal(ripple_r_walk_mean, ripple_r_walk_sd, 
                (max(market_info['Date']).to_pydatetime() - datetime.datetime.strptime(split_date, '%Y-%m-%d')).days + 1)

fig, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax2.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y')  for i in range(12)])
ax3.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax3.set_xticklabels('')
ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
     market_info[market_info['Date']>= split_date]['bt_Close'].values, label='Actual')
ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
      market_info[(market_info['Date']+ datetime.timedelta(days=1))>= split_date]['bt_Close'].values[1:] * 
     (1+bt_random_steps), label='Predicted')
ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
     market_info[market_info['Date']>= split_date]['eth_Close'].values, label='Actual')
ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
      market_info[(market_info['Date']+ datetime.timedelta(days=1))>= split_date]['eth_Close'].values[1:] * 
     (1+eth_random_steps), label='Predicted')

ax3.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
     market_info[market_info['Date']>= split_date]['ripple_Close'].values, label='Actual')
ax3.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
      market_info[(market_info['Date']+ datetime.timedelta(days=1))>= split_date]['ripple_Close'].values[1:] * 
     (1+ripple_random_steps), label='Predicted')


ax1.set_title('Single Point Random Walk (Test Set)')
ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
ax2.set_ylabel('Ethereum Price ($)',fontsize=12)
ax3.set_ylabel('Ripple Price ($)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
plt.tight_layout()
plt.show()


# In[46]:

bt_random_walk = []
eth_random_walk = []
ripple_random_walk = []
for n_step, (bt_step, eth_step, ripple_step) in enumerate(zip(bt_random_steps, eth_random_steps, ripple_random_steps)):
    if n_step==0:
        bt_random_walk.append(market_info[market_info['Date']< split_date]['bt_Close'].values[0] * (bt_step+1))
        eth_random_walk.append(market_info[market_info['Date']< split_date]['eth_Close'].values[0] * (eth_step+1))
        ripple_random_walk.append(market_info[market_info['Date']< split_date]['ripple_Close'].values[0] * (ripple_step+1))
    else:
        bt_random_walk.append(bt_random_walk[n_step-1] * (bt_step+1))
        eth_random_walk.append(eth_random_walk[n_step-1] * (eth_step+1))
        ripple_random_walk.append(ripple_random_walk[n_step-1] * (ripple_step+1))
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax2.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y')  for i in range(12)])
ax3.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax3.set_xticklabels('')

ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date']>= split_date]['bt_Close'].values, label='Actual')
ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         bt_random_walk[::-1], label='Predicted')
ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date']>= split_date]['eth_Close'].values, label='Actual')
ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         eth_random_walk[::-1], label='Predicted')

ax3.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date']>= split_date]['ripple_Close'].values, label='Actual')
ax3.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         ripple_random_walk[::-1], label='Predicted')

ax1.set_title('Full Interval Random Walk')
ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
ax2.set_ylabel('Ethereum Price ($)',fontsize=12)
ax3.set_ylabel('Ripple Price ($)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
plt.tight_layout()
plt.show()


# In[47]:

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


# In[49]:

#ripple not integrated

def plot_func(freq):
    np.random.seed(freq)
    random_steps = np.random.normal(eth_r_walk_mean, eth_r_walk_sd, 
                (max(market_info['Date']).to_pydatetime() - datetime.datetime.strptime(split_date, '%Y-%m-%d')).days + 1)
    random_walk = []
    for n_step,i in enumerate(random_steps):
        if n_step==0:
            random_walk.append(market_info[market_info['Date']< split_date]['eth_Close'].values[0] * (i+1))
        else:
            random_walk.append(random_walk[n_step-1] * (i+1))
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
    ax1.set_xticklabels('')
    ax2.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
    ax2.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y')  for i in range(12)])
    ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date']>= split_date]['eth_Close'].values, label='Actual')
    ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
          market_info[(market_info['Date']+ datetime.timedelta(days=1))>= split_date]['eth_Close'].values[1:] * 
         (1+random_steps), label='Predicted')
    ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
          market_info[(market_info['Date']+ datetime.timedelta(days=1))>= split_date]['eth_Close'].values[1:] * 
         (1+random_steps))
    ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
             random_walk[::-1])
    ax1.set_title('Single Point Random Walk')
    ax1.set_ylabel('')
    # for static figures, you may wish to insert the random seed value
#    ax1.annotate('Random Seed: %d'%freq, xy=(0.75, 0.2),  xycoords='axes fraction',
#            xytext=(0.75, 0.2), textcoords='axes fraction')
    ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
    ax2.set_title('Full Interval Random Walk')
    fig.text(0.0, 0.5, 'Ethereum Price ($)', va='center', rotation='vertical',fontsize=12)
    plt.tight_layout()
#    plt.savefig('image%d.png'%freq, bbox_inches='tight')
    plt.show()
    
interact(plot_func, freq =widgets.IntSlider(min=200,max=210,step=1,value=205, description='Random Seed:'))


# In[50]:

#LSTM


# In[51]:

for coins in ['bt_', 'eth_','ripple_']: 
    kwargs = { coins+'close_off_high': lambda x: 2*(x[coins+'High']- x[coins+'Close'])/(x[coins+'High']-x[coins+'Low'])-1,
            coins+'volatility': lambda x: (x[coins+'High']- x[coins+'Low'])/(x[coins+'Open'])}
    market_info = market_info.assign(**kwargs)


# In[52]:

model_data = market_info[['Date']+[coin+metric for coin in ['bt_', 'eth_','ripple_'] 
                                   for metric in ['Close','Volume','close_off_high','volatility']]]
# need to reverse the data frame so that subsequent rows represent later timepoints
model_data = model_data.sort_values(by='Date')
model_data.head()


# In[53]:

# we don't need the date columns anymore
training_set, test_set = model_data[model_data['Date']<split_date], model_data[model_data['Date']>=split_date]
training_set = training_set.drop('Date', 1)
test_set = test_set.drop('Date', 1)


# In[54]:

window_len = 10
norm_cols = [coin+metric for coin in ['bt_', 'eth_', 'ripple_'] for metric in ['Close','Volume']]


# In[64]:

LSTM_training_inputs = []
for i in range(len(training_set)-window_len):
    temp_set = training_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_training_inputs.append(temp_set)
LSTM_training_outputs_1 = (training_set['eth_Close'][window_len:].values/training_set['eth_Close'][:-window_len].values)-1
LSTM_training_outputs_2 = (training_set['ripple_Close'][window_len:].values/training_set['ripple_Close'][:-window_len].values)-1


# In[65]:

LSTM_training_outputs_1


# In[66]:

LSTM_training_outputs_2


# In[67]:

LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
    temp_set = test_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_test_inputs.append(temp_set)
LSTM_test_outputs_1 = (test_set['eth_Close'][window_len:].values/test_set['eth_Close'][:-window_len].values)-1
LSTM_test_outputs_2 = (test_set['ripple_Close'][window_len:].values/test_set['ripple_Close'][:-window_len].values)-1


# In[68]:

LSTM_training_inputs[0]


# In[69]:

# I find it easier to work with numpy arrays rather than pandas dataframes
# especially as we now only have numerical data
LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)


# In[5]:

# import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[4]:

get_ipython().system('pip install keras')


# In[7]:

def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


# In[8]:

get_ipython().system('pip install tensorflow')


# In[10]:

get_ipython().system('pip install theano')


# In[18]:

get_ipython().system('pip install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.4-cp36-cp36m-win_amd64.whl')


# In[21]:

get_ipython().system('pip install https://cntk.ai/PythonWheel/GPU/cntk-2.4-cp36-cp36m-win_amd64.whl')


# In[22]:

get_ipython().system('pip install https://cntk.ai/PythonWheel/GPU-1bit-SGD/cntk-2.4-cp36-cp36m-win_amd64.whl')


# In[ ]:



