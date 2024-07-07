#STEP 1
from transformers import pipeline

#STEP 2
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")

#STEP 3
text = "positive"

#STEP 4
result = classifier(text)

#STEP 5
print(result)
