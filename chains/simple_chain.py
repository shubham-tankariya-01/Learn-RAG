from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_protocol import InputInject
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = init_chat_model("qwen/qwen3.8-27b",model_provider="groq")
prompt = PromptTemplate(
    template="generate 5 intersting facts about in one lines only {topic}",
    input_variables=['topic']
)
parser = StrOutputParser()

#LCEL
chain = prompt | model | parser
chain.get_graph().print_ascii()
res = chain.invoke({'topic' : 'harry potter'})

print(res)
