from abc import ABC, abstractmethod

class ResultsControllerInterface(ABC):
	@abstractmethod
	def label_results_by_id(self) -> dict: 
		raise NotImplementedError

	@abstractmethod
	def build_docs_terms(self) -> dict: 
		raise NotImplementedError

	@abstractmethod
	def collect_vocabulary(self, docs_terms:dict) -> set: 
		raise NotImplementedError

	@abstractmethod
	def vectorize(self, input_features:dict, vocabulary:list) -> dict: 
		raise NotImplementedError

	@abstractmethod
	def calculate_idf_verctors(self, docs_terms:dict, vocabulary:dict) -> dict:
		raise NotImplementedError

	@abstractmethod
	def calculate_cosine_similarity(self, query_vector:list, docs_vector:dict) -> dict:
		raise NotImplementedError

	@abstractmethod
	def sorted_results(self, results:dict) -> list:
		raise NotImplementedError

	@abstractmethod
	def sorted_docs(self, sorted_results:dict) -> list:
		raise NotImplementedError
