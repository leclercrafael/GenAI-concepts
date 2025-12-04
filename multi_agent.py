import os
from dotenv import load_dotenv
import asyncio
import google
import warnings
import logging

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from base_agent import BaseAgent

from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

console = Console()

warnings.simplefilter("ignore")
logging.basicConfig(level=logging.CRITICAL)

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError('API key was not found')
print('API key loaded')


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


class ResearchAgent(BaseAgent):

    def __init__(self):
        super().__init__()

        self.research_agent = Agent(
                                    name = 'researcher',
                                    model=Gemini(retry_options=retry_config, model='gemini-2.5-flash-lite'),
                                    instruction='You are a researcher agent, you are only goal is to ' \
                                    'find two or three informations on the given topic and present the findings with citation',
                                    tools = [google_search],
                                    output_key='research_findings'
                                    )



        self.summary_agent = Agent(
                                    name = 'summarizer',
                                    model = Gemini(retry_options=retry_config, model='gemini-2.5-flash-lite'),
                                    instruction = 'Your only goal is to summarize in 100 words the informations on a ' \
                                    'topic I am going to give to you. Here are the information : {research_findings}' ,
                                    output_key= 'summary'
                                    )
        
        self.root_agent = Agent(
                                name = 'root_agent',
                                model=Gemini(retry_options=retry_config, model='gemini-2.5-flash-lite'),
                                instruction="""You are a research coordinator. Your goal is to answer the user's query by orchestrating a workflow. \
                                1. First, you MUST call the `researcher` tool to find relevant information on the topic provided by the user.
                                2. Next, after receiving the research findings, you MUST call the `summarizer` tool to create a concise summary.
                                3. Finally, present the final summary clearly to the user as your response.""",
                                tools = [AgentTool(self.research_agent), AgentTool(self.summary_agent)],
                                )   


    async def run(self, input : str) -> str :
        
        runner = InMemoryRunner(agent=self.root_agent)
        response = await runner.run_debug(input)

        return response
    

if __name__ == "__main__":

    research_agent = ResearchAgent()

    console.rule("[bold green]Research Agent[/bold green]")
    console.print('[bold] Ask your question :[/bold]', end= " ")
    user_input = input()

    if user_input:
        with console.status("[bold green]Agents are working...[/bold green]", spinner="dots"):
            console.print("\n")
            asyncio.run(research_agent.run(user_input))


    else: 
        console.print('You did not write anything')



    



