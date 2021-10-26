# McKinley Harlett

# In[63]:


# Packages
import pandas as pd
import matplotlib.pyplot as plt
import requests 
from bs4 import BeautifulSoup
import lxml
import json
import numpy as np


# ### Source 1: Flat File

# In[64]:


states = pd.read_csv("Deaths_by_State")
states.head()


# ##### Cleaning/Prepping Steps

# In[65]:


# only want to keep data that is number of death indicator

states_death = states.loc[states['Indicator'] == "Number of Deaths"]


# In[66]:


# Now keeping data that is not labeled united states
states_death2 = states_death.loc[states['State'] != 'UNITED STATES']


# In[67]:


# keeping only 2020 data
states_death3 = states_death2.loc[states['Year'] == 2020]


# In[68]:


# only wanting to keep State, Year, Month and Data Value
final_states = states_death3[["State", "Month", "Data Value"]]


# In[69]:


final_states = final_states.rename(columns={'Data Value':'DataValue'})


# In[70]:


deaths = pd.DataFrame(data=final_states)
deaths.head()


# ###### Graph

# In[71]:


fig, ax = plt.subplots()
plt.bar(deaths['Month'], deaths['DataValue'])
plt.show()


# ### Source 2: Website Data

# In[72]:


#Website URL
url = 'https://en.wikipedia.org/wiki/Statistics_of_the_COVID-19_pandemic_in_the_United_States'
#Read the wiki site
html = pd.read_html(url)


# In[73]:


#The number of tables
print('Number of tables on the site: ',len(html))


# In[74]:


#Copy the table to a dataframe and preview it.
df = html[1]
#Print the table below
df.head()


# In[13]:


# Dropping unwanted columns
df.drop(df.columns[[0,6,7,8]], axis = 1, inplace = True)


# In[14]:


# Getting rid of the last header
df.columns = df.columns.droplevel(-1)


# In[15]:


df.columns=df.columns.get_level_values(1)


# In[16]:


# Once again, cleaning up the header
df = df.rename(columns={'U.S. state or territory[i]':'State_Territory', 'Cases[ii]':'Cases', 'Deaths':'Deaths', 'Recov.[iii]':'Recovering', 'Hosp.[iv]':'Hospilized'})


# In[17]:


# Last two rows are not wanted or needed
n = 2
df.drop(df.tail(n).index,
        inplace = True)


# In[18]:


# Now getting rid of the non 50 US States
df1 = df.drop([df.index[2] , df.index[9], df.index[12], df.index[37], df.index[42], df.index[48]])


# In[23]:


df1["Recovering"].replace({"–": "0"}, inplace=True)
df1["Hospilized"].replace({"–": "0"}, inplace=True)
df1["Deaths"].replace({"6,117[vi]": "6117"}, inplace=True)
df1["Deaths"].replace({"40,635[viii]": "40635"}, inplace=True)
df1["Recovering"].replace({"2,518,534[ix]": "2518534"}, inplace=True)
df1.head()


# ##### Graph

# In[44]:


result = df1.groupby('State_Territory').max().sort_values(by='Cases', ascending=False)[:10]

result.Cases=pd.to_numeric(result.Cases)
result.Deaths=pd.to_numeric(result.Deaths)
result.Recovering=pd.to_numeric(result.Recovering)
result.Hospilized=pd.to_numeric(result.Hospilized)

result.plot(kind='bar')
plt.show()


# ### Source 3: JSON File

# In[45]:


# Opening JSON file
f = open('Population by State.json',)
  
# returns JSON object as 
# a dictionary
data = json.load(f)

for i in data:
    print(i)

# Closing file
f.close()


# In[46]:


# Making JSON into a dataframe
pop = pd.DataFrame(data)
pop.head()


# In[47]:


# I want the population as of 2021 but then I also want to compare it to 2010 and see if 
# we are growing at a large amount which means that the deaths would be more significant due to COVID
pop1 = pop[["State", "Pop", "Growth", "Pop2010", "density"]] 


# In[53]:


pop2 = pop1.drop_duplicates(subset='State', keep='first') #dropping any columns that are duplicated in State columm


# In[54]:


pop3 = pop2.drop([pop2.index[30] , pop2.index[49]]) # Dropping the rows that are Puerto Rico and District of Columbia


# In[55]:


pop3["pop_12"] = pop3["Pop"]/12


# In[56]:


pop3["pop_12"] = pop3["pop_12"].apply(lambda x: '%.1f' % x) # Changing the output from scientific notation to a readable format


# In[57]:


pop3.head()


# ##### Graph

# In[62]:


result2 = pop3.groupby('State').max().sort_values(by='Pop', ascending=False)[:10]

result2.plot(kind='bar')
plt.show()


# ## Importing into SQLite

# In[60]:


import sqlite3


# In[61]:


from pathlib import Path
Path('population.db').touch()


# In[62]:


conn = sqlite3.connect('population.db')
c = conn.cursor()


# In[63]:


c.execute('''CREATE TABLE population (State text, Pop int, Growth int, Pop2010 int, density int, pop_12 int)''')


# In[64]:


pop3.to_sql('population', conn, if_exists='append', index = False)


# In[65]:


c.execute('''SELECT * FROM population''').fetchall()


# In[66]:


Path('covid.db').touch()


# In[67]:


conn2 = sqlite3.connect('covid.db')
c2 = conn2.cursor()


# In[68]:


c2.execute('''CREATE TABLE COVIDData (State_Territory text, Cases int, Deaths int, Recovering int, Hospilized int)''')


# In[69]:


df1.to_sql('COVIDData', conn2, if_exists='append', index = False) # write to sqlite table


# In[70]:


c2.execute('''SELECT * FROM COVIDData''').fetchall()


# In[71]:


Path('death.db')


# In[72]:


conn3 = sqlite3.connect('death.db')
c3 = conn3.cursor()


# In[73]:


c3.execute('''CREATE TABLE Deaths (State text, Month text, DataValue int)''')


# In[74]:


deaths.to_sql('Deaths', conn3, if_exists='append', index = False) # write to sqlite table


# In[75]:


c3.execute('''SELECT * FROM Deaths''').fetchall()

