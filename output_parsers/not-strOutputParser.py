
import os
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

# max tokens are set because we free groq tier is allwing max token +== 1000 (output) model is generating nearly ~1400 on every specific detailed req.
model = ChatGroq(model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"), max_tokens=900)

template1 = PromptTemplate(template="write a notes on {topic} under 800 tokens",input_variables=['topic'])
template2 = PromptTemplate(template="write a 5 Line summary on the following text /n . {text}", input_variables=['text'])


prompt1 = template1.invoke({'topic' : 'blackhole'})
response = model.invoke(prompt1)
prompt2 = template2.invoke({'text' : response.content})
result = model.invoke(prompt2)

print(result.content)


# model = ChatGroq(model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"))
# text = " make a report on blackwholes"
# response = model.invoke(text)
