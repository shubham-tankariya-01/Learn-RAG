
import os
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser,ResponseSchema
from dotenv import load_dotenv
load_dotenv()

# max tokens are set because we free groq tier is allwing max token +== 1000 (output) model is generating nearly ~1400 on every specific detailed req.
model = ChatGroq(model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"), max_tokens=900)

#creating the schema with response schema , which will be exact name and value pair in output
schema = [
    ResponseSchema(name="name", description= "the name of the person/charactor"),
    ResponseSchema(name="role", description= "what is the role of that character in movie"),
    ResponseSchema(name="movie_name", description= "from all movies , in which movie did it appear first"),
]

#created a parser to parse the output
parser = StructuredOutputParser.from_response_schemas(schema)


template1 = PromptTemplate(
    template="give me the name role and movie of any fictional character from harry potter \n {formate_instruction}",
    input_variables=[],
    partial_variables={'formate_instruction' : parser.get_format_instructions()} # this will just add a instruction to follow the same json that we created while initing the parser
 )

#creating the prompt
prompt = template1.format()

print(prompt)
'''
give me the name role and age of any fictional character from harry potter
 The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":

```json
{
        "name": string  // the name of the person/charactor
        "role": string  // what is the role of that character in movie
        "movie_name": string  // from all movies , in which movie did it appear first
}
```
'''
result = model.invoke(prompt)
final_result_object=parser.parse(result.text)

print(final_result_object)
# {'name': 'Harry Potter', 'role': 'The Boy Who Lived / Protagonist', 'movie_name': "Harry Potter and the Philosopher's Stone"}

#---------------------------------------------------------------------------------------------------------
#----------------------------------------with chains ------------------------------------------------------

chain = template1 | model | parser
final_res = chain.invoke({})
print(final_res) #type is class<dict>
print(type(final_res)) #type is <class'dict'>

'''
{'name': 'Harry Potter', 'role': 'Protagonist', 'movie_name': "Harry Potter and the Philosopher's Stone"}
'''

#_______________________________________KEY TAKE_________________________________________________________
# ## 6. Important Methods

# | Method | Purpose |
# |---|---|
# | `from_response_schemas()` | Creates a parser using response schemas |
# | `get_format_instructions()` | Generates instructions for the LLM |
# | `parse(text)` | Converts formatted text into a Python dictionary |

# ---

# ## 7. Limitations / Cons of StructuredOutputParser

# This is an important section. StructuredOutputParser is useful, but it has limitations.

# ### 1. No strict schema validation

# Suppose we expect:

# ```json
# {
#     "name": "Harry Potter",
#     "role": "Protagonist",
#     "movie_name": "Harry Potter and the Philosopher's Stone"
# }
# ```

# But the LLM generates:

# ```json
# {
#     "name": "Harry Potter",
#     "role": "Protagonist"
# }
# ```

# The `movie_name` field is missing.

# StructuredOutputParser does not enforce the full schema like a strict validation library would.

# ### 2. No strict data type validation

# Suppose we expect all values to be strings.

# But the LLM returns:

# ```json
# {
#     "name": 123,
#     "role": true,
#     "movie_name": ["Movie 1", "Movie 2"]
# }
# ```

# This is valid JSON, so the parser can parse it into a Python dictionary.

# It does not strictly enforce the expected data types described in ResponseSchema.

# ### 3. Depends on LLM instructions

# The parser provides formatting instructions, but the LLM may ignore them.

# For example, it might return:

# ```text
# Here is the information you requested:

# {
#     "name": "Harry Potter",
#     "role": "Protagonist",
#     "movie_name": "Harry Potter and the Philosopher's Stone"
# }
# ```

# Depending on the response format, extra text outside the JSON may cause parsing errors.

# ### 4. Invalid JSON can cause parsing errors

# For example:

# ```json
# {
#     "name": "Harry Potter",
#     "role": "Protagonist",
# }
# ```

# The trailing comma makes this invalid JSON.

# The parser cannot reliably parse malformed JSON.

# You need separate retry or correction logic if you want automatic recovery.

# ### 5. No guarantee of semantic correctness

# Even if the JSON is perfectly valid and contains all the expected fields, the information can still be wrong.

# Example:

# ```json
# {
#     "name": "Harry Potter",
#     "role": "Villain",
#     "movie_name": "Avengers: Endgame"
# }
# ```

# The JSON structure is valid, but the information is incorrect.

# A parser checks the structure it can parse, not whether the LLM's answer is factually correct.

# ### 6. Extra fields may appear

# You define three fields, but the LLM may return an additional field:

# ```json
# {
#     "name": "Harry Potter",
#     "role": "Protagonist",
#     "movie_name": "Harry Potter and the Philosopher's Stone",
#     "house": "Gryffindor"
# }
# ```

# Do not assume that defining three ResponseSchema entries guarantees exactly three output keys
