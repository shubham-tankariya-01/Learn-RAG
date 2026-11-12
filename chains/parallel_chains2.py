
from langchain_core.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
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
# 1. SKILL EXTRACTION CHAIN
# ---------------------------------

skills_prompt = PromptTemplate(
    template="""
    Extract relevant technical skills from this resume.

    Resume:
    {resume}

    Categorize them into:
    - Programming languages
    - Frameworks
    - AI/ML technologies
    - Databases
    """,
    input_variables=["resume"]
)

skills_chain = skills_prompt | model | parser

# ---------------------------------
# 2. RESUME REVIEW CHAIN
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

review_chain = review_prompt | model | parser


# ---------------------------------
# 3. JOB MATCHING CHAIN
# ---------------------------------

matching_prompt = PromptTemplate(
    template="""
    Compare this resume with the job description.

    Resume:
    {resume}

    Job Description:
    {job_description}

    Provide:
    - Score out of 10
    - Estimated match percentage
    - Matching skills
    - Missing skills
    - Recommendations
    """,
    input_variables=["resume", "job_description"]
)

matching_chain = matching_prompt | model | parser


# ---------------------------------
# 4. PARALLEL CHAIN
# ---------------------------------

parallel_chain = RunnableParallel({
    "skills": skills_chain,
    "review": review_chain,
    "matching": matching_chain
})


# ---------------------------------
# 5. MERGE PROMPT
# ---------------------------------

merge_prompt = PromptTemplate(
    template="""
    You are an expert AI recruitment assistant.

    Create a professional resume analysis report
    using the following three analysis results.

    SKILLS EXTRACTION:
    {skills}

    RESUME REVIEW:
    {review}

    JOB MATCHING:
    {matching}

    Prepare a final report with these sections:

    1. Candidate Technical Profile
    2. Resume Strengths and Weaknesses
    3. Job Compatibility
    4. Missing Skills
    5. Actionable Improvement Plan

    Make the report clear and practical.
    Do not invent information.
    """,
    input_variables=["skills", "review", "matching"]
)


# ---------------------------------
# 6. MERGE CHAIN
# ---------------------------------

merge_chain = merge_prompt | model | parser


# ---------------------------------
# 7. SEQUENTIAL COMPOSITION
# ---------------------------------

chain = parallel_chain | merge_chain


# ---------------------------------
# 8. INVOKE
# ---------------------------------

result = chain.invoke(input_data)

print("\n========== FINAL RESUME ANALYSIS ==========\n")
print(result)
