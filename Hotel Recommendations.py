#!/usr/bin/env python
# coding: utf-8

# # Hotel Recommendations

# In[1]:


# Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import random
import numpy as np

import ml_metrics as metrics

from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')


# ## Loading Data

# In[2]:


# Limited the number of rows 
train = pd.read_csv('train.csv', nrows=18500000)


# In[3]:


dest = pd.read_csv('destinations.csv')


# ## EDA

# In[4]:


train.head()


# In[5]:


dest.head()


# #### Density Plot of Hotel Clusters

# In[6]:


# Wanting to see if there are any hotel clusters that stand out from the rest.
sns.distplot(train['hotel_cluster'])
plt.show()

# Seems to be fairly distributed. 


# #### Heat map

# In[7]:


sns.set(style="white")


# In[8]:


# Grabbing correlation
corr = train.corr()


# In[9]:


# This allows blanks for the upper part of our heat map
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


# In[10]:


# found this awesome color palette online
cmap = sns.diverging_palette(220, 10, as_cmap=True)


# In[11]:


# Draw the heatmap with the blanks and correct aspect ratio   
sns.set(font_scale=.3)

plt.figure(figsize=(5,5), dpi=2000)

sns_plot=sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[12]:


# I want to condense my data
unique_users = set(train.user_id.unique())
rand_user_id = random.sample(unique_users,10000)


# In[13]:


# Next I want to iterate it over the entire training set.
# append the rows for the 10,000 users I sampled.
# That way we preserve the full data of each user.

sample_train = pd.DataFrame()
train_chunk = pd.read_csv('train.csv', iterator = True, chunksize = 1000000)
for chunk in train_chunk:
    sample_train = sample_train.append(chunk.loc[chunk['user_id'].isin(rand_user_id)])


# ### Changing Date

# In[14]:


sample_train['date_time'] = pd.to_datetime(sample_train['date_time'])
sample_train['year'] = sample_train['date_time'].dt.year
sample_train['month'] = sample_train['date_time'].dt.month


# #### Splitting our Data

# In[15]:


train2 = sample_train[((sample_train.year == 2013) | ((sample_train.year == 2014) & (sample_train.month < 8)))]
test2 = sample_train[((sample_train.year == 2014) & (sample_train.month >= 8))]


# In[16]:


# Wanting our test data to booking data only
test2 = test2[test2.is_booking == 1]
test2.head()


# ## Algorithm 1

# #### Most common hotel clusters

# In[17]:


most_common_clusters = list(train2.hotel_cluster.value_counts().head().index)
print(most_common_clusters)


# In[18]:


# We want to use the most common clusters as our first list of predictions for each row in test2.
predictions = [most_common_clusters for i in range(test2.shape[0])]


# In[19]:


target = [[l] for l in test2['hotel_cluster']]
metrics.mapk(target, predictions, k=5)


# ##### We are looking at a 6% accuracy 
# ##### Not every good for our first prediction but at least we now know what we are working with

# ## Machine Learning

# Want just a couple features from Dest file

# In[20]:


# Found this awesome PCA package that will allow me to compress the columns

pca = PCA(n_components=3)
dest_small = pca.fit_transform(dest[["d{0}".format(i + 1) for i in range(149)]])
dest_small = pd.DataFrame(dest_small)
dest_small["srch_destination_id"] = dest["srch_destination_id"]


# ### Next while doing some background research, I found this very handy chunk of code.
# ##### The steps are the following:
# 
# ###### Generate new date features based on date_time, srch_ci (check-in date), and srch_co (check-out date)
# ###### Remove the date_time column since it's not needed  
# ###### add features from dest_small 
# ###### Replace any missing values with -1 

# In[21]:


def calc_fast_features(df):
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["srch_ci"] = pd.to_datetime(df["srch_ci"], format='%Y-%m-%d', errors="coerce")
    df["srch_co"] = pd.to_datetime(df["srch_co"], format='%Y-%m-%d', errors="coerce")
    
    props = {}
    for prop in ["month", "day", "hour", "minute", "dayofweek", "quarter"]:
        props[prop] = getattr(df["date_time"].dt, prop)
    
    carryover = [p for p in df.columns if p not in ["date_time", "srch_ci", "srch_co"]]
    for prop in carryover:
        props[prop] = df[prop]
    
    date_props = ["month", "day", "dayofweek", "quarter"]
    for prop in date_props:
        props["ci_{0}".format(prop)] = getattr(df["srch_ci"].dt, prop)
        props["co_{0}".format(prop)] = getattr(df["srch_co"].dt, prop)
    props["stay_span"] = (df["srch_co"] - df["srch_ci"]).astype('timedelta64[h]')
        
    ret = pd.DataFrame(props)
    
    ret = ret.join(dest_small, on="srch_destination_id", how='left', rsuffix="dest")
    ret = ret.drop("srch_destination_iddest", axis=1)
    return ret

df = calc_fast_features(train2)
df.fillna(-1, inplace=True)


# In[22]:


df.head()


# ## Algorithm 2: Random Forest Classifier

# In[23]:


predictors = [c for c in df.columns if c not in ["hotel_cluster"]]

clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
scores = cross_val_score(clf, df[predictors], df['hotel_cluster'], cv=3)
scores


# As expected, we can see here that we have around 7% accuracy! Darn, but onto one more!

# ## Algorithm 3: Clustering based on Destination

# ##### I would first like to point out that I did not come up with this myself 
# 
# ##### I did infact read through this and typed it all out and "made it my own"
# 
# ##### I understand what is being coded and the thought process behind it

# In[34]:


def make_key(items):
    return "_".join([str(i) for i in items])

match_cols = ["srch_destination_id"] # Grabbing Destination ID
cluster_cols = match_cols + ['hotel_cluster'] # Adding Destination ID to Hotel Cluster
groups = train2.groupby(cluster_cols) # Grouping our training data to the cluster cols
top_clusters = {}
for name, group in groups:
    clicks = len(group.is_booking[group.is_booking == False]) # Clicks are False and Assigning .15 to it
    bookings = len(group.is_booking[group.is_booking == True]) # Bookings are True and Assigning 1 to it
    
    score = bookings + .15 * clicks # getting the score based on click or booked
    
    clus_name = make_key(name[:len(match_cols)])
    if clus_name not in top_clusters:
        top_clusters[clus_name] = {}
    top_clusters[clus_name][name[-1]] = score


# In[35]:


# Looking for the top 7 hotel clusters
import operator

cluster_dict = {}
for n in top_clusters:
    tc = top_clusters[n]
    top = [l[0] for l in sorted(tc.items(), key=operator.itemgetter(1), reverse=True)[:7]] # Top 7 Hotels
    cluster_dict[n] = top


# In[36]:


preds = []
for index, row in test2.iterrows():
    key = make_key([row[m] for m in match_cols])
    if key in cluster_dict:
        preds.append(cluster_dict[key])
    else:
        preds.append([])


# In[37]:


preds[0:5]


# In[38]:


metrics.mapk([[l] for l in test2["hotel_cluster"]], preds, k=7)


# #### Look at that! We got our accuracy up to 24%
