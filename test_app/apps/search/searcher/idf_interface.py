from abc import ABC, abstractmethod

class IdfInterface(ABC):
	@abstractmethod
	def calculate_idfs(self, vocabulary:list, doc_features:dict) -> dict:
		raise NotImplementedError

	@abstractmethod
	def vectorize_idf(self, input_terms:dict, input_idfs:dict, vocabulary:list) -> dict:
		raise NotImplementedError
