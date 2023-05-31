from transformers import EncoderDecoderModel, AutoTokenizer
from arabert.preprocess import ArabertPreprocessor
from chatbot_utils import bert_get_intent, get_entities
from flask import Flask, request, jsonify
from statemachine import StateMachine, State
from recommender_utils import search, initialize, dsgr_recommend
import random
import torch
import spacy

class ChatbotMachine(StateMachine):
    "A chatbot machine"
    start = State(initial=True)
    input_ask_entities = State()
    input_ask_no_entities = State()
    get_query_based_recommendations = State()
    get_dsgr_recommendations = State()
    end = State(final=True)

    start_to_input_ask_entities = start.to(input_ask_entities)
    start_to_input_ask_no_entities = start.to(input_ask_no_entities)

    input_ask_entities_to_get_query_based_rec = input_ask_entities.to(get_query_based_recommendations)

    input_ask_no_entities_to_input_ask_entities = input_ask_no_entities.to(input_ask_entities)
    #input ask no entities to start
    re_ask_user = input_ask_no_entities.to(start)
    input_ask_no_entities_to_get_dsgr_rec = input_ask_no_entities.to(get_dsgr_recommendations)

    get_dsgr_rec_to_input_ask_entities = get_dsgr_recommendations.to(input_ask_entities)
    get_dsgr_rec_to_get_dsgr_rec = get_dsgr_recommendations.to(get_dsgr_recommendations)
    get_dsgr_rec_to_end = get_dsgr_recommendations.to(end)

    get_query_based_rec_to_end = get_query_based_recommendations.to(end)
    
    
    def __init__(self):
        self.ner_model = spacy.load("ner_model/")
        self.bert = None
        self.tokenizer = None
        self.pre = None
        self.minimum_length = 5
        self.k = 5
        self.message_to_send = None
        self.recipe = None
        self.elastic_recipes, self.cbow_recipes, self.dsgr = initialize()
        self.recommendations_index = 0
        self.recommendations = None
        super(ChatbotMachine, self).__init__()

    def get_intent(self, text):
        #use bert to get intent
        return bert_get_intent(text, self.pre, self.tokenizer, self.bert, self.minimum_length, self.k)

    def get_entities(self, text):
        # Implement your logic to get the list of entities
        return get_entities(self.ner_model, text)
    
    def before_start(self, msg):
        intent = self.get_intent(msg)
        if "Ask" in intent:
            entities = self.get_entities(msg)
            print(entities)
            if entities:
                return self.start_to_input_ask_entities()
            else:
                return self.start_to_input_ask_no_entities()

    def before_input_ask_entities(self, msg):
        entities = self.get_entities(msg)
        self.message_to_send = random.choice(ask_responses)
        self.input_ask_entities_to_get_query_based_rec()
        search_results = search(entities, self.elastic_recipes, self.cbow_recipes, msg)
        self.recipe = search_results[self.recommendations_index]
        self.recommendations = search_results
        return self.message_to_send, self.recipe

    def before_input_ask_no_entities(self, msg):
        self.message_to_send = random.choice(ask_for_query_responses)
        return self.message_to_send
        # intent = self.get_intent(msg)
        # if intent == "Agree":
        #     return self.input_ask_no_entities_to_input_ask_entities()
        # elif intent == "Refuse":
        #     return self.input_ask_no_entities_to_get_dsgr_rec()

    def after_input_ask_no_entities(self, msg):
        intent = self.get_intent(msg)
        if "Ask" in intent:
            entities = self.get_entities(msg)
            if entities:
                self.input_ask_no_entities_to_input_ask_entities()
                self.input_ask_entities_to_get_query_based_rec()
                self.message_to_send = random.choice(ask_responses)
                return self.message_to_send
            else:
                self.message_to_send = random.choice(ask_for_query_responses)
                return self.message_to_send
        elif "Agree" in intent:
            self.message_to_send = random.choice(ask_for_query_responses)
            return self.message_to_send
        elif "Refuse" in intent:
            self.recommendations = dsgr_recommend(self.dsgr, self.elastic_recipes)
            self.recommendations_index = 0
            self.recipe = self.recommendations[self.recommendations_index]
            self.message_to_send = random.choice(giving_recommendations).format(self.recipe['name'])
            self.input_ask_no_entities_to_get_dsgr_rec()
            return self.message_to_send, self.recipe
        return None
            
    def after_get_dsgr_recommendations(self):
        self.recommendations = dsgr_recommend(self.dsgr, self.elastic_recipes)
        self.recommendations_index = 0
        self.recipe = self.recommendations[self.recommendations_index]
        self.message_to_send = random.choice(giving_recommendations).format(self.recipe['name'])
        return self.message_to_send, self.recipe
            
    def before_get_dsgr_recommendations(self, msg):
        intent = self.get_intent(msg)
        if "Ask" in intent:
            self.get_dsgr_rec_to_input_ask_entities()
            self.input_ask_entities_to_get_query_based_rec()
            return self.before_get_query_based_recommendations(msg)
        elif "Refuse" in intent:
            self.get_dsgr_rec_to_get_dsgr_rec()
            self.recommendations_index += 1
            self.recipe = self.recommendations[self.recommendations_index]
            self.message_to_send = random.choice(giving_recommendations).format(self.recipe['name'])
            return self.message_to_send, self.recipe
        elif "Agree" in intent:
            self.get_dsgr_rec_to_end()
            self.message_to_send = random.choice(accept_responses)
            return self.message_to_send

    def before_get_query_based_recommendations(self, msg):
        intent = self.get_intent(msg)
        if "Agree" in intent:
            self.message_to_send = random.choice(accept_responses)
            self.get_query_based_rec_to_end()
            return self.message_to_send
        elif "Refuse" in intent:
            self.message_to_send = random.choice(refuse_responses)
            self.recommendations_index += 1
            if self.recommendations_index >= len(self.recommendations):
                #get new recommendations
                self.recommendations = search(self.get_entities(msg), self.elastic_recipes, self.cbow_recipes, msg)
                self.recommendations_index = 0
            self.recipe = self.recommendations[self.recommendations_index]
            return self.message_to_send, self.recipe
        else:
            self.message_to_send = random.choice(ask_responses)
            #make new recommendations
            self.recommendations_index = 0
            self.recommendations = search(self.get_entities(msg), self.elastic_recipes, self.cbow_recipes, msg)
            self.recipe = self.recommendations[self.recommendations_index]
            return self.message_to_send, self.recipe

    def reset(self):
        self.current_state = self.start
        

    def send_message(self, message):
        current_state = self.current_state
        if current_state == self.start:
            result = self.before_start(message)
            if self.current_state == self.input_ask_entities:
                result = self.before_input_ask_entities(message)
            elif self.current_state == self.input_ask_no_entities:
                result = self.before_input_ask_no_entities(message)
        elif current_state == self.input_ask_no_entities:
            result = self.after_input_ask_no_entities(message)
            if self.current_state == self.get_dsgr_recommendations:
                result = self.after_get_dsgr_recommendations()
        elif current_state == self.get_dsgr_recommendations:
            result = self.before_get_dsgr_recommendations(message)
        elif current_state == self.get_query_based_recommendations:
            result = self.before_get_query_based_recommendations(message)
        elif current_state == self.end:
            self.message_to_send = random.choice(accept_responses)
            result = self.message_to_send
        else:
            result = None
            
        if result:
            return result
        else:
            return "Invalid state transition."


ask_responses = ["حاضر هشوفلك", "تمام ثانية", "هدورلك اهو", "ماشي حاضر"]
refuse_responses = ["طب ثانية اجبلك حاجة تانية", "نشوف حاجة تانية"]
accept_responses = ["بالهنا و الشفا", "يلا بينا", "يلا"]
giving_recommendations = ["ايه رايك في {}", "تيجي نجرب {}", "هو ده اللي انت عاوزه {}", "هو ده اللي انت عاوزه {}"]
ask_for_query_responses = ["عاوز تدور على ايه؟", "في حاجة معينة في بالك؟"]

# Instantiate the chatbot machine
chatbot = ChatbotMachine()
print("chat bot machine created")
model_name="bert-base-arabert"
arabert_prep = ArabertPreprocessor(model_name=model_name, keep_emojis=False)
tokenizer = AutoTokenizer.from_pretrained("./arabert2arabert")
model = EncoderDecoderModel.from_pretrained("./arabert2arabert")

chatbot.bert = model
chatbot.tokenizer = tokenizer
chatbot.pre = arabert_prep
model.to("cuda")
model.eval()

ner_model = spacy.load("ner_model/")
print("done")

app = Flask(__name__)

@app.route('/chat/start', methods=['GET'])
def chat():
    if chatbot.current_state == ChatbotMachine.start:
        #send greeting message and ask for entities
        response = "هلا، أنا بوت الطبخ، ممكن أساعدك في البحث عن وصفات طبخ. عاوز تدور على أي نوع من الوصفات؟"
        return response

#route chat/msg to get response
@app.route('/chat', methods=['POST'])
def chatbot_api():
    data = request.json
    message = data.get('message', '')

    # Send the message to the chatbot
    response = chatbot.send_message(message)
    print(response)
    #if we got reponse type tuple, then we have a recipe to recommend 
    #and we need to send the recipe name and the response
    if isinstance(response, tuple):
        return jsonify({'response': response[0], 'recipe': response[1]})
    return jsonify({'response': response})
        
@app.route('/chat/reset', methods=['GET'])
def reset():
    chatbot.reset()
    return "Reset done"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=205)