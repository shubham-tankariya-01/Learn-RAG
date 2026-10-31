import os

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()


# Pydantic model:
# This defines the exact fields and data types expected from the LLM.
# PydanticOutputParser validates the model response against this schema.
class Character(BaseModel):
    name: str = Field(description="The name of the fictional character")
    role: str = Field(description="The role of the character in the movie")
    movie_name: str = Field(description="The first Harry Potter movie in which the character appeared")


# max_tokens is set because the free Groq tier supports a limited output size.
model = ChatGroq(
    model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
    max_tokens=900,
)

# Create a parser using the Pydantic model.
parser = PydanticOutputParser(pydantic_object=Character)

# get_format_instructions() generates instructions that tell the LLM
# which fields and JSON format it must return.
template1 = PromptTemplate(
    template=(
        "Give me the name, role, and first movie of any fictional character "
        "from Harry Potter.\n{format_instructions}"
    ),
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Print the final prompt sent to the model.
prompt = template1.format()
print(prompt)

# Manual parsing:
# First call the model, then parse its text response.
result = model.invoke(prompt)
final_result = parser.parse(result.text)
print(final_result)
# output :-
# name='Harry Potter' role='Protagonist'
# movie_name="Harry Potter and the Philosopher's Stone"

# ---------------------------------------------------------------------------
# Parsing with a chain:
# The prompt, model, and parser are connected into one runnable pipeline.
chain = template1 | model | parser
final_result_from_chain = chain.invoke({})
print(final_result_from_chain)
# output :-
# name='Harry Potter' role='Protagonist'
# movie_name="Harry Potter and the Philosopher's Stone"


# IMPORTANT METHODS
#
# | Method | Purpose |
# |---|---|
# | `PydanticOutputParser()` | Creates a parser from a Pydantic model |
# | `get_format_instructions()` | Generates the required output instructions |
# | `parse(text)` | Validates text and returns a Pydantic object |


# ADVANTAGES OF PydanticOutputParser
#
# 1. STRICT FIELD VALIDATION
#    The response is expected to contain the fields defined in Character.
#
# 2. DATA TYPE VALIDATION
#    Pydantic checks that each field has the expected type.
#    For example, `name: str` must contain text.
#
# 3. PYDANTIC OBJECT OUTPUT
#    The result is a Character object instead of an unstructured string
#    or a plain dictionary.
#
#    final_result.name
#    final_result.role
#    final_result.movie_name


# LIMITATIONS / CONS OF PydanticOutputParser
#
# 1. THE LLM STILL HAS TO FOLLOW THE FORMAT
#    Format instructions guide the LLM, but the model can still return
#    invalid JSON or unexpected text. In that case, parsing raises an error.
#
# 2. REQUIRED FIELDS MUST BE PRESENT
#    The fields in Character are required by default. A missing field causes
#    Pydantic validation to fail.
#
# 3. TYPES MUST MATCH THE MODEL
#    If a response cannot be converted to the declared field type,
#    Pydantic validation fails.
#
# 4. SCHEMA CHANGES REQUIRE MODEL CHANGES
#    If the required output changes, update the Pydantic model and its fields.


# WHEN TO USE:
#
# PydanticOutputParser:
# - When the response must follow a known schema.
# - When required fields and data types need validation.
# - When the application should work with a typed Python object.
#
# JsonOutputParser:
# - When flexible JSON converted into dictionaries or lists is enough.
#
# StrOutputParser:
# - When only the model's plain text response is required.


# KEY TAKEAWAY:
# PydanticOutputParser converts the LLM response into a validated Pydantic
# object, making structured output safer and easier to use in Python code.
