from langchain_core.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# Model
model = ChatGroq(
    model="qwen/qwen3.8-27b",
    max_tokens=900
)

parser = StrOutputParser()

# ---------------------------------
# INPUT DATA
# ---------------------------------

resume = """
I am a computer science student.
I have experience with Python, JavaScript, React,
Node.js, Express, MongoDB and FastAPI.
I have built a meeting application using React,
FastAPI and LiveKit.
I have basic knowledge of LangChain and RAG.
"""

job_description = """
We are hiring an AI Engineer.

Required skills:
Python, FastAPI, LangChain, RAG,
LLMs, Vector Databases and SQL.

Candidates should understand backend development
and have experience building AI applications.
"""


input_data = {
    "resume": resume,
    "job_description": job_description
}


# ---------------------------------
# 1. SKILL EXTRACTION PROMPT
# ---------------------------------

skills_prompt = PromptTemplate(
    template="""
    Extract all relevant technical skills from this resume.

    Resume:
    {resume}

    Return categorized skills:
    - Programming languages
    - Frameworks
    - AI/ML technologies
    - Databases
    """,
    input_variables=["resume"]
)


# ---------------------------------
# 2. RESUME REVIEW PROMPT
# ---------------------------------

review_prompt = PromptTemplate(
    template="""
    Analyze this resume.

    Resume:
    {resume}

    Identify:
    - Strengths
    - Weaknesses
    - Missing information
    - Improvements needed
    """,
    input_variables=["resume"]
)


# ---------------------------------
# 3. JOB MATCHING PROMPT
# ---------------------------------

matching_prompt = PromptTemplate(
    template="""
    Compare this resume with the job description.

    Resume:
    {resume}

    Job Description:
    {job_description}

    Provide:
    - score out of 10
    - Estimated match percentage
    - Matching skills
    - Missing skills
    - Recommendations
    """,
    input_variables=["resume", "job_description"]
)


# ---------------------------------
# CREATE INDIVIDUAL CHAINS
# ---------------------------------

skills_chain = skills_prompt | model | parser

review_chain = review_prompt | model | parser

matching_chain = matching_prompt | model | parser


# ---------------------------------
# PARALLEL EXECUTION
# COMBINE ORIGINAL INPUT + OUTPUTS
# ---------------------------------


chain = RunnablePassthrough.assign(
    skills=skills_chain,
    review=review_chain,
    matching=matching_chain
)


# ---------------------------------
# FINAL OUTPUT
# ---------------------------------

result = chain.invoke(input_data)


print("\n========== SKILLS ==========")
print(result["skills"])

print("\n========== RESUME REVIEW ==========")
print(result["review"])

print("\n========== JOB MATCHING ==========")
print(result["matching"])

chain.get_graph().print_ascii()
