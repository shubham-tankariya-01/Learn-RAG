
from langchain_core.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableBranch , RunnableLambda
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Literal
load_dotenv()

# Model
model = ChatGroq(
    model="qwen/qwen3.8-27b",
    max_tokens=900
)

class FeedBack(BaseModel):
    sentiment : Literal['positive' , 'negative'] = Field(description="it is the sentiment of the feedback can be either postive or negative")

parser = StrOutputParser()
parserpy = PydanticOutputParser(pydantic_object=FeedBack)

prompt1 = PromptTemplate(
    template="classify the following sentiment of the user feedback text into positive or negative \n {feedback} \n {format_instruction}",
        partial_variables={'format_instruction' : parserpy.get_format_instructions()},
        input_variables=['feedback']
        )
prompt2 = PromptTemplate(
    template="write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template="write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)

classifier_chain = prompt1 | model | parserpy
chain2 = prompt2 | model | parser
chain3 = prompt3 | model | parser

branch_chain = RunnableBranch(
    (lambda x : x.sentiment  == 'positive', chain2),
    (lambda x : x.sentiment == 'negative', chain3),
    RunnableLambda( lambda x : "not able to find sentiment")
)

chain = classifier_chain | branch_chain | parser

response = chain.invoke({'feedback' : 'This is terrible phone , functions are not working properly'})

print(response)
