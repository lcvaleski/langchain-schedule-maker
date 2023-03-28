import os
import streamlit as st
from pathlib import Path
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from llama_index import GPTSimpleVectorIndex, download_loader, SimpleDirectoryReader

# NOTE: for local testing only, do NOT deploy with your key hardcoded
# to use this for yourself, create a file called .streamlit/secrets.toml with your api key
# Learn more about Streamlit on the docs: https://docs.streamlit.io/

os.environ["OPENAI_API_KEY"] = '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'
llm = OpenAI(temperature=0, openai_api_key="OPENAI_API_KEY")

index = GPTSimpleVectorIndex.load_from_disk('index.json')

st.header("Schedule generator")

text = st.text_input("Enter course titles:")

if st.button("Run Query") and text is not None:
    courses = (index.query(
    "List all " + text + "classes"
    ))

    st.markdown(courses)
    schedule_template = """

    When a student is registering for courses, there are very often many different possible schedules that they
    could sign up for. The reason for this is that there are multiple of the same course at times, each with their own times and meeting days.

    The student would like to compare and contrast different schedule plans, given the courses they need to take.
    Walk the student through all of the different options for their weekly shedules in an easy to read and friendly manner.

    If there is a course that has multiple class options, create an entirely new schedule for the student. List all of the options out for the student
    so they can make a decision about which is best for them.

    If there are conflicting times for a given schedule, explain to the user.

    Example input and output:

    Input: 

    Seeing Stories: Reading Race and Graphic Narratives Seminar CRN: 50135 Instructor: Sohn, Stephen (Primary) Meeting days and times: Monday,Thursday, 02:30 PM - 03:45 PM

    eeing Stories: Reading Race and Graphic Narratives Seminar CRN: 50136 Instructor: Meeting days and times: Monday,Thursday, 04:00 PM - 05:15 PM

    African American Philosophy Lecture CRN: 49937 Instructor: Green, Judith (Primary) Meeting days and times: Tuesday,Friday, 01:00 PM - 02:15 PM  


    Output:
    Schedule 1:

    Seeing Stories: Reading Race and Graphic Narratives Seminar
    Meeting days and times: Monday, Thursday 02:30 PM - 03:45 PM; Tuesday, Friday 01:00 PM - 02:15 PM
    African American Philosophy Lecture
    Meeting days and times: Monday, Tuesday, Thursday, Friday 01:00 PM - 02:15 PM

    Schedule 2:

    Seeing Stories: Reading Race and Graphic Narratives Seminar
    Meeting days and times: Monday, Thursday 04:00 PM - 05:15 PM; Tuesday, Friday 01:00 PM - 02:15 PM
    African American Philosophy Lecture
    Meeting days and times: Monday, Tuesday, Thursday, Friday 01:00 PM - 02:15 PM

    {courses}

    schedule:
    """

    schedule_prompt = PromptTemplate(template=schedule_template, input_variables=["courses"])
    schedule_chain = LLMChain(llm=llm, prompt=schedule_prompt)
    schedule = schedule_chain.run(courses)
    st.markdown(schedule)
