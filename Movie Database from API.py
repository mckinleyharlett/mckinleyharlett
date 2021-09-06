#!/usr/bin/env python
# coding: utf-8

# # Activity 10

# ## Building your own movie database by reading from an API

# In[1]:


import urllib.request, urllib.parse, urllib.error
import json


# #### Load the secret API key (you have to get one from OMDB website and use that, 1000 daily limit) from a JSON file, stored in the same folder into a variable
# Hint: Use json.loads()
# 
# Note: The following cell will not be executed in the solution notebook because the author cannot give out his private API key.
# Students/users/instructor will need to obtain a key and store in a JSON file.
# For the code's sake, we are calling this file APIkeys.json. But you need to store your own key in this file.
# An example file called "APIkey_Bogus_example.json" is given along with the notebook. Just change the code in this file and rename as APIkeys.json. The file name does not matter of course.

# In[10]:


# we want to open the key and load it in as the key
with open('mckinley.json') as f:
    keys = json.load(f)
    omdbapi = keys['OMDBapi']


# #### The final URL to be passed should look like: http://www.omdbapi.com/?t=movie_name&apikey=secretapikey
# Do the following,
# 
# Assign the OMDB portal (http://www.omdbapi.com/?) as a string to a variable serviceurl (don't miss the ?)
# Create a variable apikey with the last portion of the URL ("&apikey=secretapikey"), where secretapikey is your own API key (an actual code)
# The movie name portion i.e. "t=movie_name" will be addressed later

# In[11]:


# this is so we are building the api key
serviceurl = 'http://www.omdbapi.com/?'
apikey = '&apikey='+omdbapi


# #### Write a utility function print_json to print nicely the movie data from a JSON file (which we will get from the portal)
# Here are the keys of a JSON file,
# 
# 'Title', 'Year', 'Rated', 'Released', 'Runtime', 'Genre', 'Director', 'Writer', 'Actors', 'Plot', 'Language','Country', 'Awards', 'Ratings', 'Metascore', 'imdbRating', 'imdbVotes', 'imdbID'

# In[5]:


# This function is used so it can be printed nicely
def print_json(json_data):
    list_keys=['Title', 'Year', 'Rated', 'Released', 'Runtime', 'Genre', 'Director', 'Writer', 
               'Actors', 'Plot', 'Language', 'Country', 'Awards', 'Ratings', 
               'Metascore', 'imdbRating', 'imdbVotes', 'imdbID'] # the information that we want from the API
    print("-"*50)
    for k in list_keys:
        if k in list(json_data.keys()):
            print(f"{k}: {json_data[k]}") # Getting the data for that specific key
    print("-"*50)


# #### Write a utility function to download a poster of the movie based on the information from the jason dataset and save in your local folder
# Use os module
# The poster data is stored in the JSON key 'Poster'
# You may want to split the name of the Poster file and extract the file extension only. Let's say the extension is 'jpg'.
# Then later join this extension to the movie name and create a filename like movie.jpg
# Use the Python command open to open a file and write the poster data. Close the file after done.
# This function may not return anything. It just saves the poster data as an image file.

# In[6]:


# This part I was AMAZED about! 
# We can download a poster of our movie we select in our local folder!

def save_poster(json_data):
    import os
    title = json_data['Title']
    poster_url = json_data['Poster']
    # Splits the poster url by '.' and picks up the last string as file extension
    poster_file_extension=poster_url.split('.')[-1]
    # Reads the image file from web
    poster_data = urllib.request.urlopen(poster_url).read()
        
    savelocation=os.getcwd()+'\\'+'Posters'+'\\'
    # Creates new directory if the directory does not exist. Otherwise, just use the existing path.
    if not os.path.isdir(savelocation):
        os.mkdir(savelocation)
    
    filename=savelocation+str(title)+'.'+poster_file_extension
    f=open(filename,'wb')
    f.write(poster_data)
    f.close()


# #### Write a utility function search_movie to search a movie by its name, print the downloaded JSON data (use the print_json function for this) and save the movie poster in the local folder (use save_poster function for this)
# Use try-except loop for this i.e. try to connect to the web portal, if successful proceed but if not (i.e. exception raised) then just print an error message
# Here use the previously created variables serviceurl and apikey
# You have to pass on a dictionary with a key t and the movie name as the corresponding value to urllib.parse.urlencode() function and then add the serviceurl and apikey to the output of the function to construct the full URL
# This URL will be used for accessing the data
# The JSON data has a key called Response. If it is True, that means the read was successful. Check this before processing the data. If not successful, then print the JSON key Error, which will contain the appropriate error message returned by the movie database.

# In[7]:


# This function is made so we can choose our title
def search_movie(title):
    try:
        url = serviceurl + urllib.parse.urlencode({'t': str(title)})+apikey
        # URL is very important bc it will give us the title we want
        print(f'Retrieving the data of "{title}" now... ')
        print(url)
        uh = urllib.request.urlopen(url)
        data = uh.read()
        json_data=json.loads(data)
        # used in case a movie is not found (url doesn't work)
        if json_data['Response']=='True':
            print_json(json_data)
            # Asks user whether to download the poster of the movie
            if json_data['Poster']!='N/A':
                save_poster(json_data)
        else:
            print("Error encountered: ",json_data['Error'])
    
    except urllib.error.URLError as e:
        print(f"ERROR: {e.reason}")


# #### Test search_movie function by entering Titanic

# In[8]:


search_movie("Titanic")


# #### Test search_movie function by entering "Random_error" (obviously this will not be found and you should be able to check whether your error catching code is working properly)

# In[9]:


search_movie("Random_error")


# In[ ]:




