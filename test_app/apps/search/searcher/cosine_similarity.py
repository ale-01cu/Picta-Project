from .coisine_simirarity_interface import CosineSimilarityInterface
import math

class CosineSimilarity(CosineSimilarityInterface):
	def length(self, vector:list) -> float | int:
		sq_length:int = 0
		for index in range(0, len(vector)):
			sq_length += math.pow(vector[index], 2)
		return math.sqrt(sq_length)

	def dot_product(self, vector1:list, vector2:list) -> float | str:
		if len(vector1)==len(vector2):
			dot_prod:int = 0
			for index in range(0, len(vector1)):
				if not vector1[index]==0 and not vector2[index]==0:
					dot_prod += vector1[index]*vector2[index]
			return dot_prod
		else:
			return "Unmatching dimensionality"

	def calculate_cosine(self, query_vector:list, doc_vector:list) -> float | str:
		cosine:float = self.dot_product(query_vector, doc_vector) / ((self.length(query_vector) * self.length(doc_vector)) + 1)
		return cosine