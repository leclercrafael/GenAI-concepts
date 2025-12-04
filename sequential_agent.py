import os
from dotenv import load_dotenv
import asyncio
import google
import warnings
import logging

from abstract_agent import AbstractAgent

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

load_dotenv()

console = Console()


if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError('API key was not found')
print('API key loaded')


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

class mySequentialAgent(AbstractAgent):

    def __init__(self) -> None :
        super().__init__()


        self.OutlineAgent = Agent(
                                    name='OutlineAgent',
                                    model= Gemini(retry_options=retry_config, model='gemini-2.5-flash-lite'),
                                    instruction='''
                                                You have to create a blog outline on the given subjetct with : 
                                                1. A catchy headline
                                                2. An introduction hook
                                                3. 3-5 main sections with 2-3 bullet points for each
                                                4. A concluding thought
                                                ''',
                                    tools = [google_search],
                                    output_key='blog_outline'
        )

        
        self.WriterAgent = Agent(
                                    name ='WriterAgent',
                                    model= Gemini(retry_options=retry_config, model='gemini-2.5-flash-lite'),
                                    instruction='''
                                                Following this outline strictly: {blog_outline}
                                                Write a brief, 200 to 300-word blog post with an engaging and informative tone.
                                                ''',
                                    output_key='blog_draft'
        )

        self.EditorAgent = Agent(
                                    name = 'EditorAgent',
                                    model= Gemini(retry_options=retry_config, model='gemini-2.5-flash-lite'),
                                    instruction='''
                                                Edit this draft: {blog_draft}
                                                Your task is to polish the text by fixing any grammatical errors, improving the flow 
                                                and sentence structure, and enhancing overall clarity.
                                                ''',
                                    output_key='final_blog'
        )



        self.RootAgent = SequentialAgent(name = 'BlogPipeline', sub_agents =[self.OutlineAgent, self.WriterAgent, self.EditorAgent])

    async def run(self, input : str) -> str:
        
        runner = InMemoryRunner(agent=self.RootAgent)
        response = await runner.run_debug(input)

        return response
    
if __name__ == "__main__":

    research_agent = mySequentialAgent()

    console.rule("[bold green]Research Agent[/bold green]")
    console.print('[bold] Ask your question :[/bold]', end= " ")
    user_input = input()

    if user_input:
        with console.status("[bold green]Agents are working...[/bold green]", spinner="dots"):
            console.print("\n")
            asyncio.run(research_agent.run(user_input))


    else: 
        console.print('You did not write anything')