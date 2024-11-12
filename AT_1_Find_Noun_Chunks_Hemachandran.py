# Databricks notebook source
dbutils.widgets.text("input_sentence", "The quick brown fox jumps over the lazy dog and happily chases a squirrel.", "Input Sentence")
input_sentence = dbutils.widgets.get("input_sentence")

# COMMAND ----------

print("The input Sentence for finding Noun Chunks is:", input_sentence)

# COMMAND ----------

# MAGIC %md
# MAGIC **Using NLTK**

# COMMAND ----------

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser

# COMMAND ----------

# Step 1: Tokenize the text
tokens = word_tokenize(input_sentence)

# Step 2: POS tagging
tagged_tokens = pos_tag(tokens)

# Step 3: Display the results
nouns = ', '.join(word for word, pos in tagged_tokens if pos == 'NN' or pos == 'NP')

print("Nouns:", nouns)

# COMMAND ----------

# MAGIC %md
# MAGIC **Using Spacy**

# COMMAND ----------

!python -m spacy download en_core_web_sm

# COMMAND ----------

import spacy
nlp = spacy.load('en_core_web_sm')

# COMMAND ----------

# Step 1: Tokenize the text
chunks = nlp(input_sentence)

# Step 2: Display the results
nouns = ', '.join(chunk.text for chunk in chunks if chunk.pos_ == "NOUN")

print("Nouns:", nouns)

# COMMAND ----------


