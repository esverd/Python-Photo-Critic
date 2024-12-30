# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Crew, Task, Process
from langchain_groq import ChatGroq

import os 
from dotenv import load_dotenv

load_dotenv()

# from langchain_groq import ChatGroq
llm_sverd = ChatGroq(
    api_key = os.getenv("GROQ_API_KEY"),
    # model = "llama-3.1-70b-versatile",
    # model = "llama3-groq-70b-8192-tool-use-preview",
    # model = "llama-3.2-90b-vision-preview",
    model = "llama-3.1-8b-instant",
    # model = "llama3-8b-8192",
    temperature = 0.4
       )

# from crewai_tools import VisionTool
# vision_tool = VisionTool()



# task_review_image = Task(
#     description=(
#         """Extract text from the provided image file. Ensure that the extracted text is accurate and complete,
#         and ready for any further analysis or processing tasks. The image file provided may contain
#         various text elements, so it's crucial to capture all readable text."""
#     ),
#     expected_output="""A string containing the full text extracted from the image.""",
#     tools=[vision_tool],
#     agent=agent_image_extractor
# )



test_crew = Crew(
    agents=[agent_image_extractor],
    tasks=[task_hello_world],
    process=Process.sequential,
    # manager_llm = llm_bore,
    # manager_agent = manager_agent,  #optional: explicitly set a specific agent as manager instead of the manager_llm
    cache = True,
    memory = True,
    embedder=dict(
            provider="ollama",
            config=dict(
                model="nomic-embed-text",
            ),
        ),
    output_log_file="runtime-log.txt",
    max_rpm = 2     #fix to not get rate limited by groq api
)


result = test_crew.kickoff()
print(result)