
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[2]:


import nltk
nltk.download('punkt')


# In[4]:


import pandas as pd
import numpy as np

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)
print(moby_tokens[:10])
print(text1)


# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[5]:


def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[6]:


def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[8]:


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[9]:


def answer_one():
    
    
    return example_two()/example_one()

answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[11]:


from nltk.probability import FreqDist
def answer_two():    
    dist = FreqDist(moby_tokens)
    whale_freq = dist['whale'] + dist['Whale']
    
    return (whale_freq/example_one())*100

answer_two()


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*

# In[12]:


def answer_three():
    return FreqDist(moby_tokens).most_common(20)

answer_three()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return an alphabetically sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*

# In[13]:


def answer_four():
    return sorted([token for token, freq in FreqDist(moby_tokens).items() 
                   if len(token) > 5 and freq > 150])

answer_four()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[14]:


def answer_five():
    
    max_length = 0
    
    for word in text1:
        word_length = len(word)
        if word_length > max_length:
            max_length = word_length
            longest_word = word
    
    return (longest_word, max_length) # Your answer here
answer_five()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[15]:


def answer_six():
    
    return sorted([(freq, word) for word, freq in FreqDist(moby_tokens).items() 
                   if word.isalpha() and freq > 2000], 
                  reverse=True)

answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[16]:


def answer_seven():
    
    num_tokens_per_sentences = [len(nltk.word_tokenize(sent)) for sent 
                                in nltk.sent_tokenize(moby_raw)]
    
    return np.mean(num_tokens_per_sentences)

answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[17]:


def answer_eight():
    tags = []
    nltk.download('averaged_perceptron_tagger')
    for sent in nltk.sent_tokenize(moby_raw):
        # apply POS on each sentence
        tagged_sent = nltk.pos_tag(nltk.word_tokenize(sent))
        
        for item in tagged_sent:
            # appending tags to the tags list
            tags.append(item[1])
            
    pos_freq = FreqDist(tags).most_common(5)
    
    return pos_freq

answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[18]:


from nltk.corpus import words

nltk.download('words')
correct_spellings = words.words()
correct_spellings[:10]


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[19]:


def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    recommended_words = []
    
    for entry in entries:
        filtered_correct_spellings = [word for word in correct_spellings if word[0] == entry[0]]
        distances = [nltk.jaccard_distance(set(nltk.ngrams(word, 3)), set(nltk.ngrams(entry, 3))) for word in 
                     filtered_correct_spellings]
        recommended_words.append(filtered_correct_spellings[np.argmin(distances)])
            
    return recommended_words
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[20]:


def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    recommended_words = []
    
    for entry in entries:
        filtered_correct_spellings = [word for word in correct_spellings if word[0] == entry[0]]
        distances = [nltk.jaccard_distance(set(nltk.ngrams(word, 4)), set(nltk.ngrams(entry, 4))) for word in 
                     filtered_correct_spellings]
        recommended_words.append(filtered_correct_spellings[np.argmin(distances)])
    
    return recommended_words
    
answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[21]:


def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    recommended_words = []
    
    for entry in entries:
        filtered_correct_spellings = [word for word in correct_spellings if word[0] == entry[0]]
        distances = [nltk.edit_distance(word, entry, transpositions=True) for word in 
                     filtered_correct_spellings]
        recommended_words.append(filtered_correct_spellings[np.argmin(distances)])
    return recommended_words 
    
answer_eleven()


# In[ ]:




