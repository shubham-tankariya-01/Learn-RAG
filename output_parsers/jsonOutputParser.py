import os
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()

# max tokens are set because we free groq tier is allwing max token +== 1000 (output) model is generating nearly ~1400 on every specific detailed req.
model = ChatGroq(model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"), max_tokens=900)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template="give me the name role and age of any fictional character from harry potter \n {formate_instruction}",
    input_variables=[],
    partial_variables={'formate_instruction' : parser.get_format_instructions()}
 )

#creating the prompt
prompt = template1.format()
#model calling
result = model.invoke(prompt)
#parsing the ouput
final_res = parser.parse(result.text) #res.text is the type of langchain textAccesor which enventually always returing the string

#------------------------------------------------***---------------------------------------------------------------------------

chain = template1 | model | parser
final_result = chain.invoke({})
print(f"final result but with chains : {final_result}")

#------------------------------------------------------------------------------------------------------------------------------

'''---------------------------------------------------------------------------------------------------------------------------'''
print(prompt)
# output :- give me the name role and age of any fictional character from harry potter \n  Return a JSON object.

print(result)
 # output ;-
'''content='```json\n{\n  "name": "Albus Dumbledore",\n  "role": "Headmaster of Hogwarts School of Witchcraft and Wizardry",\n  "age": 115\n}\n```' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 34, 'total_tokens': 80, 'completion_time': 0.087432245, 'completion_tokens_details': None, 'prompt_time': 0.002336332, 'prompt_tokens_details': None, 'queue_time': 0.079306618, 'total_time': 0.089768577}, 'model_name': 'qwen/qwen3.8-27b', 'system_fingerprint': 'fp_424cb89518', 'service_tier': 'on_demand', 'finish_reason': 'stop', 'logprobs': None, 'model_provider': 'groq'} id='lc_run--01a1914d-b514-7681-b6a5-3cb63959edeb-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 34, 'output_tokens': 46, 'total_tokens': 80}'''

print(result.content)
print(result.text)
#output of both :-
'''
{
  "name": "Harry Potter",
  "role": "Main protagonist / The Boy Who Lived",
  "age": 11
}
'''

print(type(result.content))
# <class 'str'>
print(type(result.text))
# <class 'langchain_core.messages.base.TextAccessor'

final_res = parser.parse(str(result.content)) # res.content can be either of str or list[str] or list[dict] , and parse only accepts the string so it is required to convert it in to a string and then passing it ; but if the content will be either of list it is not possible to correct it from str() becuase it will  return a string representation of list , that may not be a valid JSON
final_res = parser.parse(result.text) #res.text is the type of langchain textAccesor which enventually always returing the string

#soo,
"""### `result.content` vs `result.text`

**`result.content`**
- Can return a `str` or a list of multimodal content blocks.
- Its type is represented as `str | list[str | dict]`.
- `parser.parse()` expects a string.
- Avoid blindly using `str(result.content)` because it may stringify a list rather than extract its text.

**`result.text`**
- Uses LangChain's `TextAccessor`.
- Provides a text-oriented interface to the message content.
- Can be passed to text-based output parsers.
- Useful when you specifically need the textual response.

**Remember:** `str(value)` converts an object to its string representation; it does not necessarily extract meaningful text from that object."""


print(final_res)
# output:-
'''
{
  "name": "Harry Potter",
  "role": "Main Character / The Boy Who Lived",
  "age": 17
}
'''


print(type(final_res))
# output :-  <class 'dict'>


# LIMITATIONS / CONS OF JsonOutputParser

# 1. NO STRICT SCHEMA VALIDATION
#    It doesn't enforce a predefined structure or schema.
#
#    Expected:
#    {
#        "name": "Harry Potter",
#        "age": 11
#    }
#
#    LLM Output:
#    {
#        "name": "Harry Potter",
#        "age": "eleven"
#    }
#
#    The parser can still parse this because "eleven" is valid JSON.
#    It doesn't know that age should be an integer.


# 2. DOESN'T GUARANTEE REQUIRED FIELDS
#    It doesn't ensure that all expected fields are present.
#
#    Expected:
#    {
#        "name": "Harry Potter",
#        "role": "Main Character",
#        "age": 11
#    }
#
#    LLM Output:
#    {
#        "name": "Harry Potter",
#        "age": 11
#    }
#
#    The parser can successfully parse it even though "role" is missing.


# 3. NO STRICT DATA TYPE VALIDATION
#    It doesn't ensure that values have the expected data types.
#
#    LLM Output:
#    {
#        "name": 123,
#        "age": "eleven",
#        "role": True
#    }
#
#    This is valid JSON, so the parser can convert it into a Python
#    dictionary without enforcing our application's expected types.


# 4. DEPENDS ON LLM INSTRUCTIONS
#    get_format_instructions() guides the LLM to generate JSON,
#    but doesn't guarantee that the LLM will follow those instructions.
#
#    The LLM might return:
#    Here is your answer:
#    {
#        "name": "Harry Potter",
#        "age": 11
#    }
#    Hope this helps!
#
#    Extra text outside JSON can cause parsing errors.


# 5. NO AUTOMATIC ERROR CORRECTION
#    If the LLM generates invalid JSON, parsing can fail.
#
#    Invalid JSON:
#    {
#        "name": "Harry Potter",
#        "age": 11,
#    }
#
#    Notice the trailing comma.
#    JsonOutputParser doesn't automatically ask the LLM to fix it.
#    We need to implement retry or correction logic separately.


# WHEN TO USE:
#
# JsonOutputParser:
# - When you need flexible JSON output converted into Python
#   dictionaries or lists.
# - When strict schema validation isn't necessary.
#
# PydanticOutputParser:
# - When you need predefined fields and expected data types.
# - When you need validation using Pydantic models.


# KEY TAKEAWAY:
# JsonOutputParser converts JSON into Python objects,
# but doesn't provide strict schema or type validation.
