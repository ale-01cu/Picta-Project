from .idf_interface import IdfInterface
import math

class Idf(IdfInterface):
	weight: dict

	def __init__(self, weight:dict) -> None:
		self.weight = weight

	def calculate_idfs(self, vocabulary:list, doc_features:dict) -> dict:
		doc_idfs:dict = {}
		doc_counts:dict = {}
		for term in vocabulary:
			doc_counts[term] = 0
			for doc_id in doc_features.keys():
				terms:list = doc_features.get(doc_id)
				if term in terms.keys():
					doc_counts[term] += terms.get(term)
					if term in self.weight:
						doc_counts[term] += 1  # Añadir 1 si el término tiene peso
					else:
						doc_counts[term] += 0.01  # Añadir un pequeño valor para evitar divisiones por cero
						doc_idfs[term] = math.log(
							float(len(doc_features.keys())) / float(doc_counts[term]), 10)
		return doc_idfs


	def vectorize_idf(self, input_terms:dict, input_idfs:dict, vocabulary:list) -> dict:
		output:dict = {}
		for item_id in input_terms.keys():
			terms:list = input_terms.get(item_id)
			output_vector:list = []
			for term in vocabulary:
				if term in terms.keys():
					output_vector.append(
					input_idfs.get(term)*float(terms.get(term)))
				else:
					output_vector.append(float(0))
			output[item_id] = output_vector
		return output