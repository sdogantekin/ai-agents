# Warning control
import warnings
warnings.filterwarnings('ignore')

import os
import yaml
import json
from crewai import Crew, Task, Agent, LLM
from ApiKeys import get_deepseek_api_key
from CustomTrelloTool import BoardDataFetcherTool, CardDataFetcherTool

files = {
    'agents': 'ProjectProgressReport/config/agents.yaml',  
    'tasks': 'ProjectProgressReport/config/tasks.yaml'
}

configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r') as file:
        configs[config_type] = yaml.safe_load(file)

agents_config = configs['agents']
tasks_config = configs['tasks']

deepseelLLM = LLM(model='deepseek/deepseek-chat', temperature=1.0, api_key=get_deepseek_api_key())

# Creating Agents
data_collection_agent = Agent(
  config=agents_config['data_collection_agent'],
  tools=[BoardDataFetcherTool(), CardDataFetcherTool()],
  llm=deepseelLLM
)

analysis_agent = Agent(
  config=agents_config['analysis_agent'],
  llm=deepseelLLM
)

# Creating Tasks
data_collection = Task(
  config=tasks_config['data_collection'],
  agent=data_collection_agent
)

data_analysis = Task(
  config=tasks_config['data_analysis'],
  agent=analysis_agent
)

report_generation = Task(
  config=tasks_config['report_generation'],
  agent=analysis_agent,
)

# Creating Crew
crew = Crew(
  agents=[
    data_collection_agent,
    analysis_agent
  ],
  tasks=[
    data_collection,
    data_analysis,
    report_generation
  ],
  verbose=True
)

result = crew.kickoff()

##cost estimation
import pandas as pd
cost_per_token = 0.015
costs = cost_per_token * (crew.usage_metrics.prompt_tokens + crew.usage_metrics.completion_tokens) / 1_000_000
print(f"Total costs: ${costs:.4f}")

from IPython.display import Markdown

markdown  = result.raw
Markdown(markdown)