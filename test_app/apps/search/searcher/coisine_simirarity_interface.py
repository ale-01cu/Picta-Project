from abc import ABC, abstractmethod

class CosineSimilarityInterface(ABC):
	@abstractmethod
	def length(self, vector:list) -> float | int:
		raise NotImplementedError

	@abstractmethod
	def dot_product(self, vector1:list, vector2:list) -> float | str:
		raise NotImplementedError

	@abstractmethod
	def calculate_cosine(self, query_vector:list, doc_vector:list) -> float | str:
		raise NotImplementedError
