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

from abstract_agent import AbstractAgent

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

class RefinementAgent(AbstractAgent):

    def __init__(self):
        super().__init__()

        self.initial_writer_agent = Agent(
            name="InitialWriterAgent",
            model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
            instruction="""Based on the user's prompt, write the first draft of a short story (around 100-150 words).
            Output only the story text, with no introduction or explanation.""",
            output_key="current_story",  # Stores the first draft in the state.
        )

        self.critic_agent = Agent(
            name="CriticAgent",
            model=Gemini(model="gemini-2.5-flash-lite",retry_options=retry_config),
            instruction="""You are a constructive story critic. Review the story provided below.
            Story: {current_story}
            
            Evaluate the story's plot, characters, and pacing.
            - If the story is well-written and complete, you MUST respond with the exact phrase: "APPROVED"
            - Otherwise, provide 2-3 specific, actionable suggestions for improvement.""",
            output_key="critique",  # Stores the feedback in the state.
        )

        self.refiner_agent = Agent(
        name="RefinerAgent",
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        instruction="""You are a story refiner. You have a story draft and critique.
        
        Story Draft: {current_story}
        Critique: {critique}
        
        Your task is to analyze the critique.
        - IF the critique is EXACTLY "APPROVED", you MUST call the `exit_loop` function and nothing else.
        - OTHERWISE, rewrite the story draft to fully incorporate the feedback from the critique.""",
        output_key="current_story",  
        tools=[FunctionTool(self.exit_loop)],  
        )

        self.story_refinement_loop = LoopAgent(
            name="StoryRefinementLoop",
            sub_agents=[self.critic_agent, self.refiner_agent],
            max_iterations=5,  # Prevents infinite loops
        )

        self.root_agent = SequentialAgent(
            name="StoryPipeline",
            sub_agents=[self.initial_writer_agent, self.story_refinement_loop],
        )

    @staticmethod
    def exit_loop() -> dict :
        return {"status": "approved", "message": "Story approved. Exiting refinement loop."}
        
    
    async def run(self, input : str) -> str:
        
        runner = InMemoryRunner(agent=self.root_agent)
        response = await runner.run_debug(input)

        return response
    
if __name__ == "__main__":

    research_agent = RefinementAgent()

    console.rule("[bold green]Refinement Agent[/bold green]")
    console.print('[bold] Ask your question :[/bold]', end= " ")
    user_input = input()

    if user_input:
        with console.status("[bold green]Agents are working...[/bold green]", spinner="dots"):
            console.print("\n")
            asyncio.run(research_agent.run(user_input))


    else: 
        console.print('You did not write anything')