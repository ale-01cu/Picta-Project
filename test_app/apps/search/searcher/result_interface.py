from abc import ABC, abstractmethod

class ResultInterface(ABC):
	@abstractmethod
	def get_id_tagged_content(self) -> dict:
		raise NotImplementedError	    

	@abstractmethod
	def get_terms(self) -> dict:
		raise NotImplementedError
		      
