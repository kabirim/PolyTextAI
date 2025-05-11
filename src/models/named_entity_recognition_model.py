import pandas as pd 
import spacy 
import requests 
from bs4 import BeautifulSoup
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
# pd.set_option("display.max_rows", 200)

def get_named_entity_recongnition(content):
    doc = nlp(content)
    # for ent in doc.ents:
    #     print(ent.text, ent.start_char, ent.end_char, ent.label_)
    # displacy.render(doc, style="ent")
    entities = [(ent.text, ent.label_, ent.lemma_) for ent in doc.ents]
    df = pd.DataFrame(entities, columns=['text', 'type', 'lemma'])
    return df

if __name__ == '__main__':
   content = "Trinamool Congress leader Mahua Moitra has moved the Supreme Court against her expulsion from the Lok Sabha over the cash-for-query allegations against her. Moitra was ousted from the Parliament last week after the Ethics Committee of the Lok Sabha found her guilty of jeopardising national security by sharing her parliamentary portal's login credentials with businessman Darshan Hiranandani."
   get_named_entity_recongnition(content)