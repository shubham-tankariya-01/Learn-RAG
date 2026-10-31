from tempfile import template

from attr import field
from langchain_core.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()

model = ChatGroq(model="qwen/qwen3.8-27b",max_tokens=900)

class movie_description(BaseModel):
    name : str = Field(description="Name of the character ")
    movie_name : str = Field(description=" name of the movie from which the character is choosen, if in all movie it exists name the best one")
    age : int = Field(description="age of the harry potter's character in the specfied movie that is choose in this output")
    scene : str = Field(description=" an in detailed description of the best scene of that movie choosen in this object , full explanation")

class scene_summary(BaseModel):
    scene_sum : list[str] = Field(description="list of 5 strings containing the summary of the provided scene")

parser = PydanticOutputParser(pydantic_object=movie_description)
parser_final_output = PydanticOutputParser(pydantic_object=scene_summary)

template1 = PromptTemplate(
    template="give a {movie} movies' information in best possible way \n {format_instructions}",
    input_variables=['movie'],
    partial_variables={'format_instructions' : parser.get_format_instructions()},
)

template2 = PromptTemplate(
    template="from given response of LLM {response} describe the scene in 5 in sentences in best possible way \n {format_instructions}",
    input_variables=['response'],
    partial_variables={'format_instructions' : parser_final_output.get_format_instructions()},
)

chain_half = template1 | model | parser
chain_full = template1 | model | parser | template2 | model | parser_final_output

response_half = chain_half.invoke({'movie' : 'all harry potter movies'})
response_full = chain_full.invoke({'movie' : 'all harry potter movies'})

print(f"this is the half response : { response_half } ")
print("\n\n\n======================================================NEXT==============================\n\n\n")
print(f"this is the full response : {response_full} ")

print("\n\n\n======================================================NEXT==============================\n\n\n")

chain_full.get_graph().print_ascii()
