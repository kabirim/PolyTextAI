# importing the BART model and tokenizer 
from transformers import BartForConditionalGeneration, BartTokenizer

# loading the pretrained weights for BART
# here, we use facebook's bart-large model
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large",
                                                     forced_bos_token_id=0) # takes a while to load
# loading the raw text tokenizer for the BART model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

def generate_automatic_text_completion(sentence):
 
    # preprocessing(tokenizing) the text as input for the BART model 
    tokenized_sent = tokenizer(sentence, return_tensors="pt")

    # generated encoded ids
    generated_encoded = bart_model.generate(tokenized_sent['input_ids'])

    # decoding the generated encoded ids
    prediction = tokenizer.batch_decode(generated_encoded, skip_special_tokens=True)[0]
    return prediction

if __name__ == '__main__':
    sent = "GeekforGeeks has a <mask> article on BART."
    generate_automatic_text_completion(sent)