from .query import Query
from .result import Result
from .idf import Idf
from .cosine_similarity import CosineSimilarity
from .results_controller_interface import ResultsControllerInterface
from .utils import merge_dict
from operator import itemgetter

class ResultsController(ResultsControllerInterface):
	query: Query
	weight: dict
	results: list
	idf: Idf
	cosine_similarity: CosineSimilarity

	def __init__(self, query:Query, results:list, weight:dict) -> None:
		self.query = query
		self.weight = weight
		self.results = []
		self.idf = Idf(self.weight)
		self.cosine_similarity = CosineSimilarity()

		for res in results:
			self.results.append(
				Result(
					res.to_dict(), 
					self.weight
				)
			)

	def label_results_by_id(self) -> dict:
		results_tagged_by_id:dict = {}

		for res in self.results:
			content_result: dict = res.get_id_tagged_content()
			new_results_tagged_by_id: dict = merge_dict(
				results_tagged_by_id, 
				content_result
			)
			results_tagged_by_id = new_results_tagged_by_id

		return results_tagged_by_id


	def build_docs_terms(self) -> dict:
		docs_terms:dict = {}

		for res in self.results:
			content_terms:dict = {res.content['id']: res.get_terms()}
			new_docs_terms:dict = merge_dict(
				docs_terms,
				content_terms
			)
			docs_terms = new_docs_terms
		return docs_terms


	def collect_vocabulary(self, docs_terms:dict) -> set:
		all_terms: dict = []
		for doc_id in docs_terms.keys():
			for term in docs_terms.get(doc_id).keys():
				all_terms.append(term)
		
		for term in self.query.query_terms.get("1").keys():
			all_terms.append(term)

		return sorted(set(all_terms))


	def vectorize(self, input_features:dict, vocabulary:list) -> dict:
		output:dict = {}
		for item_id in input_features.keys():
			features = input_features.get(item_id)
			output_vector:list = []
			for word in vocabulary:
				if word in features.keys():
					output_vector.append(int(features.get(word)))
				else:
					output_vector.append(0)
			output[item_id] = output_vector
		return output


	def calculate_idf_verctors(self, docs_terms:dict, vocabulary:dict) -> dict:
		docs_idfs:dict = self.idf.calculate_idfs(
			vocabulary, 
			docs_terms
		)

		docs_idfs_vectors:dict = self.idf.vectorize_idf(
			docs_terms, 
			docs_idfs, 
			vocabulary
		)

		return docs_idfs_vectors


	def calculate_cosine_similarity(self, query_vector:list, docs_vector:dict) -> dict:
		result:dict = {}
		for doc_id in docs_vector.keys():
			document:list = docs_vector.get(doc_id)
			cosine:float = self.cosine_similarity.calculate_cosine(
				query_vector, 
				document
			)
			result[doc_id] = cosine

		return result


	def sorted_results(self, results:dict) -> list:
		return sorted(
			results.items(), 
			key=itemgetter(1), 
			reverse=True
		)[:20]


	def sorted_docs(self, sorted_results:dict) -> list:
		label_results_by_id:dict = self.label_results_by_id()
		return [label_results_by_id[i[0]] for i in sorted_results]
	    