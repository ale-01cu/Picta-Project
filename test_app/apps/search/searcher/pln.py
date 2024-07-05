from nltk import word_tokenize, regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
import string
import re

class PLN:
	stop_list_spanish: set
	stop_list_english: set
	pattern: str
	stemmer: None

	def __init__(self) -> None:
		self.stop_list_spanish = set(stopwords.words('spanish'))
		self.stop_list_english = set(stopwords.words('english'))
		self.pattern = r'\w+|[^\w\s]'
		self.stemmer = SnowballStemmer('spanish')


	def get_processed_terms(self, text:str) -> dict:
		local_terms:dict = {}

		word_list = [
			self.stemmer.stem(word.lower()) for word in regexp_tokenize(text, self.pattern)
			if word.isalnum() 
			and word.lower() not in self.stop_list_spanish 
			and word.lower() not in self.stop_list_english 
			and word.lower() not in string.punctuation
			]

		for word in word_list:
			local_terms[word] = local_terms.get(word, 0) + 1

		return local_terms