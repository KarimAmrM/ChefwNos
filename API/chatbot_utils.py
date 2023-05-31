def bert_get_intent(text, arabert_prep, tokenizer, model,minimum_length, k):
  text_clean = arabert_prep.preprocess(text)
  inputs = tokenizer.encode_plus(text_clean,return_tensors='pt')
  outputs = model.generate(input_ids = inputs.input_ids.to("cuda"),
                   attention_mask = inputs.attention_mask.to("cuda"),
                   num_beams=1,
                   do_sample = True,
                   min_length=minimum_length,
                   top_k = k,
                   temperature = 1,
                   length_penalty =2)
  preds = tokenizer.batch_decode(outputs) 
  response = str(preds)
  response = response.replace("\'", '')
  response = response.replace("[[CLS]", '')
  response = response.replace("[SEP]]", '')
  print(response)
  response = str(arabert_prep.desegment(response))
  #get first word
  response = response.split()[0]
  #keep only english letters
  response = ''.join(filter(str.isalpha, response))
  return response


def get_entities(model, msg):
    #get entities from text and tokens
    doc = model(msg)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return entities