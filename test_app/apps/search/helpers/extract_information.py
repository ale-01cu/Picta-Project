import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("On Friday, board members meet with senior managers " +
"to discuss future development of the company.")

for chunk in doc.noun_chunks:
	print('\t'.join([chunk.text, chunk.root.text, chunk.root.dep_,
		chunk.root.head.text]))



from spacy import displacy
from pathlib import Path

svg = displacy.render(doc, style='dep', jupyter=False)
file_name = '-'.join([w.text for w in doc if not w.is_punct]) + ".svg"
output_path = Path(file_name)
output_path.open("w", encoding="utf-8").write(svg)

for token in doc:
	print(token.text, token.dep_,
		token.head.text, token.head.pos_,
		[child for child in token.children])


for token in doc:
	if (token.lemma_=="meet" and token.pos_=="VERB" and token.dep_=="ROOT"):
		action = token.text
		children = [child for child in token.children]
		participant1 = ""
		participant2 = ""

		for child1 in children:
			if child1.dep_=="nsubj":
				participant1 = " ".join(
					[attr.text for attr in child1.children]
				) + " " + child1.text
			elif child1.text=="with":
				action += " " + child1.text
				child1_children = [child for child in child1.children]
				for child2 in child1_children:
					if child2.pos_ == "NOUN":
						participant2 = " ".join(
							[attr.text for attr in child2.children]
						) + " " + child2.text

sentences = ["On Friday, board members meet with senior managers " +
"to discuss future development of the company.",
"Boris Johnson met with the Queen last week.","Donald Trump meets the Queen at Buckingham Palace.",
"The two leaders also posed for photographs and " +
"the President talked to reporters."]



def extract_information(doc):
	action=""
	participant1 = ""
	participant2 = ""
	for token in doc:
		if (token.lemma_=="meet" and token.pos_=="VERB"
			and token.dep_=="ROOT"):
			action = token.text
			children = [child for child in token.children]
			for child1 in children:
				if child1.dep_=="nsubj":
					participant1 = " ".join(
						[attr.text for attr in child1.children]
					) + " " + child1.text
				elif child1.text=="with":
					action += " " + child1.text
					child1_children = [child for child in child1.children]
					for child2 in child1_children:
						if (child2.pos_ == "NOUN"
							or child2.pos_ == "PROPN"):
							participant2 = " ".join(
								[attr.text for attr in child2.children]
							) + " " + child2.text
				elif (child1.dep_=="dobj"
					and (child1.pos_ == "NOUN"
						or child1.pos_ == "PROPN")):
					participant2 = " ".join(
						[attr.text for attr in child1.children]
					) + " " + child1.text

	print (f"Participant1 = {participant1}")
	print (f"Action = {action}")
	print (f"Participant2 = {participant2}")


for sent in sentences:
	print(f"\nSentence = {sent}")
	doc = nlp(sent)
	extract_information(doc)