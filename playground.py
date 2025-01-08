from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k

from phi.playground import Playground, serve_playground_app

from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
phi_api_key=os.getenv("PHI_API_KEY")

import phi
import groq 


phi.api_key = phi_api_key 
groq.api_key = groq_api_key

## Research

Research_agent = Agent(
    model=Groq(id="llama-3.2-11b-vision-preview"),
    tools=[DuckDuckGo(), Newspaper4k()],
    description="You are a senior NYT researcher writing an article on a topic.",
    instructions=[
        "For a given topic, search for the top 2 links.",
        "Then read each URL and extract the article text, if a URL isn't available, ignore it.",
        "Analyse and prepare an NYT worthy article based on the information.",
    ],
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    # debug_mode=True,
)

app=Playground(agents=[Research_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)