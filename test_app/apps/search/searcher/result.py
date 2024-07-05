from .pln import PLN

class Result:
	id: int | str
	content: dict
	weighing: dict
	result_terms: dict
	text_procesor: PLN

	def __init__(self, content:dict, weighing:dict) -> None:
		self.id = content['id']
		self.content = content
		self.weighing = weighing
		self.result_terms = {}
		self.text_procesor = PLN()

	def get_id_tagged_content(self) -> dict:
		aux:dict = {} 
		aux[self.content['id']] = self.content
		return aux

	def get_terms(self) -> dict:
		terms: dict = {}
		for key, value in self.get_id_tagged_content()[self.id].items():
			if key in self.weighing:
				weigh: float | int = self.weighing[key]
				value_terms: dict = self.text_procesor.get_processed_terms(value)
				for term, count in value_terms.items():
					terms[term] = terms.get(term, 0) + count * weigh
				
		self.result_terms[self.id] = terms
		return terms