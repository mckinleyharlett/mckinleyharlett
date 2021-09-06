#!/usr/bin/env python
# coding: utf-8

# # Activity 7

# ## Creating a soup with bs4 and loading the data

# In[38]:


from bs4 import BeautifulSoup
import pandas as pd


# ### Task 1

# In[39]:


# We are wanting to open, read, store and then close the web file
fd = open("List of countries by GDP (nominal) - Wikipedia.html", "r")


# In[40]:


soup = BeautifulSoup(fd, 'html.parser')
fd.close()


# ### Task 2: How many tables are in our data?

# In[48]:


print(soup)


# In[41]:


# We use find_all because it will allow us to look at the number of tables within soup
all_tables = soup.find_all("table")

# We can use {} to insert the number of tables without having to break the ""
print("Number of Tables is {}".format(len(all_tables)))


# ### Task 3: Finding the right table using the class attribute

# In[52]:


# I needed to just grab the tables and their class was simply wikitable not wikitable|}
data_table = soup.find('table',{'class':'wikitable'})
print(type(data_table))


# ### Task 4: Separating the source from the actual data

# In[51]:


# tr opens the table 
# td ads the data cell

# Which is why we want to find the open/close table and then pull the cell data


sources = data_table.tbody.findAll('tr', recursive = False)[0]

sources_list = [td for td in sources.findAll('td')]

print(len(sources_list))


# In[53]:


# Now we want to find the actual data, above was the sources now this is the actual data that we want.
data = data_table.tbody.findAll('tr', recusursive = False)[1].findAll('td', recursive = False)


# In[55]:


data_tables = []
for td in data:
    data_tables.append(td.findAll('table'))


# In[59]:


len(data_tables)
print(data_tables)


# ### Task 5: Getting the Source Names

# In[57]:


# To get the names we just have to find a because that will give us the source names
source_names = [source.findAll('a')[0].getText() for source in sources_list]
print(source_names)


# ### Task 6: Separate Headers and Data into a nice table for all 3 Sources

# #### International Monetary

# In[62]:


# Looking at just the headers by finding the first row and column the th
header_IM = [th.getText().strip() for th in data_tables[0][0].findAll('th')]
header_IM


# In[64]:


# Now we want to find the rows so we use [1:] as that indicator

rows_IM = data_tables[0][0].findAll('tbody')[0].findAll('tr')[1:]


# In[65]:


# Now from those rows we want to strip the data! 

data_rows_IM = [[td.get_text().strip() for td in tr.findAll('td')] for tr in rows_IM]


# In[66]:


df_IM = pd.DataFrame(data_rows_IM, columns = header_IM)


# In[67]:


df_IM.head()


# ##### World Bank

# In[68]:


# We are going to be doing the same steps as above
header_WB = [th.getText().strip() for th in data_tables[1][0].findAll('th')]
header_WB


# In[69]:


rows_WB = data_tables[1][0].findAll('tbody')[0].findAll('tr')[1:]


# In[70]:


# This is different because we have a specific symbol that we need to get rid of 
def find_right_text(i, td):
    if i == 0:
        return td.getText().strip()
    elif i == 1:
        return td.getText().strip()
    else:
        index = td.text.find("♠")
        return td.text[index+1:].strip()


# In[71]:


data_rows_WB = [[find_right_text(i,td) for i, td in enumerate(tr.findAll('td'))] for tr in rows_WB]


# In[72]:


df_WB = pd.DataFrame(data_rows_WB, columns = header_WB)


# In[73]:


df_WB.head()


# #### United Nations

# In[76]:


header_UN = [th.getText().strip() for th in data_tables[2][0].findAll('th')]
header_UN


# In[77]:


rows_UN = data_tables[2][0].findAll('tbody')[0].findAll('tr')[1:]


# In[78]:


data_rows_UN = [[find_right_text(i,td) for i, td in enumerate(tr.findAll('td'))] for tr in rows_UN]


# In[79]:


df_UN = pd.DataFrame(data_rows_UN, columns = header_UN)


# In[80]:


df_UN.head()

