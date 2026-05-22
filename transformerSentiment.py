from transformers import pipeline

nlp_llm = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
print(nlp_llm("I am saying this restaurant is good"))
