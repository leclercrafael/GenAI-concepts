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

class ParallelAgent(AbstractAgent);
    
    def __init__(self):
        super().__init__()

        self.TechResearcher = Agent(
                                    name='TechResearcher',
                                    model=Gemini(retry_options=retry_config, model='gemini-2.5-flash-lite'),
                                    instruction='''
                                    Research the latest AI/ML trends. Include 3 key developpements,
                                    the main companies involved and the impact. Keep it short (100 words)
                                    ''',
                                    tools=[google_search],
                                    output_key='tech_news'
        )

        self.HealthResearcher = Agent(
                                    name='HealthResearcher',
                                    model=Gemini(retry_options=retry_config, model='gemini-2.5-flash-lite'),
                                    instruction='''
                                    Research the latest medical trends, and breakthroughs. Include 3 key developpements,
                                    the main applications and the impact it could have. Keep it short (100 words)
                                    ''',
                                    tools=[google_search],
                                    output_key='health_news'
        )

        self.FinanceResearcher = Agent(
                                    name='FinanceResearcher',
                                    model=Gemini(retry_options=retry_config, model='gemini-2.5-flash-lite'),
                                    instruction='''
                                    Research the current finance trends and breakthroughs. Include 3 key trends,
                                    the market applications and the impact it could have. Keep it short (100 words)
                                    ''',
                                    tools=[google_search],
                                    output_key='finance_news'
        )

        self.AgregatorAgent = Agent(
                                    name='AgregatorAgent',
                                    model=Gemini(retry_options=retry_config, model='gemini-2.5-flash-lite'),
                                    instruction='''
                                    Combine these three research findings into a single executive summary ;
                                    Tech news : {tech_news}

                                    Health news : {health_news}
                                    
                                    Finance news : {finance_news}
                                    ''',
                                    tools=[google_search],
                                    output_key='finance_news'
        )