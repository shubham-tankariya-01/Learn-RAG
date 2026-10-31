
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
    max_tokens=900
)

template1 = PromptTemplate(
    template="Write notes on {topic} under 800 tokens.",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5-line summary of the following text:\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser| RunnableLambda(lambda text  : {'text' : text}) | template2 | model | parser

result = chain.invoke({"topic": "Black Holes"})

print(result)
