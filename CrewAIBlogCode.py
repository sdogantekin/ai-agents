## sample crew ai flow for the Medium blog post
## for more details, check --> https://github.com/sdogantekin/ai-agents
from crewai import Crew, Task, Agent, LLM
from ApiKeys import get_deepseek_api_key
from crewai_tools import ScrapeWebsiteTool
from CustomDuckDuckGoTool import CustomDuckDuckGoTool

# in this example, we use the deepseek-chat model, but you can use any other model you have access to (even the local ones)
deepseekllm = LLM(model='deepseek/deepseek-chat',temperature=1.0,api_key=get_deepseek_api_key())
# llamaLLM = LLM(model="ollama/llama3.2",base_url="http://localhost:11434") # sample reference to local LLama model

# these are tools that we will assign to the agents
scrape_tool = ScrapeWebsiteTool() # this tool will enable an agent to scrape a given website
search_tool = CustomDuckDuckGoTool() # this tool will enable an agent to search the web for a given query

# this the agent that will analyze the job description
job_analysis_specialist = Agent(
    role='Job Analysis Specialist',
    goal='Analyze job descriptions thoroughly to extract key requirements and company profile',
    backstory="You are an experienced job market analyst with deep understanding of " 
        "various industries and roles. You excel at breaking down job descriptions and target company profiles "
        "to identify both explicit and implicit requirements.",
    verbose=True,
    allow_delegation=False,
    tools=[scrape_tool, search_tool],
    llm=deepseekllm
)

# this the agent that will analyze the candidate profile
profile_analysis_specialist = Agent(
    role='Profile Analysis Specialist',
    goal='Analyze candidate profiles to identify strengths, weaknesses, and unique selling points',
    backstory="You are a career coach with years of experience in helping professionals "
        "present their experience effectively. You excel at identifying transferable "
        "skills and unique value propositions.",
    verbose=True,
    allow_delegation=False,
    tools=[scrape_tool, search_tool],
    llm=deepseekllm
)

# this the agent that will provide matching advice
career_matching_specialist = Agent(
    role='Career Matching Specialist',
    goal='Evaluate job-candidate fit and provide strategic application advice',
    backstory="You are a senior career advisor who specializes in helping candidates "
        "position themselves effectively for roles. You excel at identifying gaps "
        "and suggesting concrete steps for improvement.",
    verbose=True,
    allow_delegation=False,
    llm=deepseekllm
)

# this is the task that will analyze the job description, this task is assigned to the job_analyzer agent
analyze_job = Task(
    description="Analyze the job posting at {job_url} and also the company related to it."
        "Focus on: "
        "1. Key technical requirements "
        "2. Soft skills and cultural fit indicators "
        "3. Implicit requirements and nice-to-haves "
        "4. Company business objectives, products, target market segment, market position, customers, success stories, values and culture signals "
        "Provide a structured analysis that can be used by the matching specialist. ",
    expected_output="Structured analysis of job requirements and target company including necessary "
        "skills, qualifications experiences and company profile including its culture.",
    agent=job_analysis_specialist
)

# this is the task that will analyze the candidate profile, this task is assigned to the profile_analyzer agent
analyze_profile = Task(
    description="Analyze the candidate profile at {profile_url}."
        "Focus on: "
        "1. Technical skills, experience and education "
        "2. Demonstrated soft skills and personality traits "
        "3. Career progression and achievements "
        "4. Unique selling points and personal brand "
        "Provide a structured analysis that can be used by the matching specialist.",
    expected_output="Structured analysis of candidate profile including key skills, experiences, contributions, interests and "
        "communication style.", 
    agent=profile_analysis_specialist
)

# this is the task that will provide matching advice, this task is assigned to the matching_specialist agent
provide_matching_advice = Task(
    description="Using the analyses from both the job and profile specialists:"
        "1. Evaluate the overall match percentage "
        "2. Identify key strengths to emphasize "
        "3. Point out gaps that need addressing "
        "4. Provide specific recommendations for: "
        "- Resume adjustments "
        "- Cover letter points "
        "- Interview preparation "
        "- Skill development priorities "
        "Provide actionable advice that the candidate can implement immediately.",
    expected_output="Detailed matching advice including strengths, gaps and "
        "recommendations for resume, cover letter, interview preparation and skill development.",
    context=[analyze_job, analyze_profile],
    agent=career_matching_specialist
)

# this is our crew that will handle the all process
application_crew = Crew(
    agents=[job_analysis_specialist, profile_analysis_specialist, career_matching_specialist],
    tasks=[analyze_job, analyze_profile, provide_matching_advice],
    verbose=True
)

# this will make our crew to work on the given tasks using the given inputs
result = application_crew.kickoff(
        inputs={
            "job_url": "https://www.linkedin.com/jobs/view/4136748557", # job posting url
            "profile_url": "https://www.linkedin.com/in/serkandogantekin" # candidate profile url
        }
    )