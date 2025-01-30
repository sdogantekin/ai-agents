## sample crea ai flow for the Medium blog post
from crewai import Crew, Task, Agent, LLM
from ApiKeys import get_deepseek_api_key
from crewai_tools import ScrapeWebsiteTool

deepseekllm = LLM(model='deepseek/deepseek-chat',temperature=1.0,api_key=get_deepseek_api_key())
scrape_tool = ScrapeWebsiteTool()

job_analyzer = Agent(
    role='Job Analysis Specialist',
    goal='Analyze job descriptions thoroughly to extract key requirements and company culture',
    backstory="You are an experienced job market analyst with deep understanding of " 
        "various industries and roles. You excel at breaking down job descriptions "
        "to identify both explicit and implicit requirements.",
    verbose=True,
    allow_delegation=False,
    tools=[scrape_tool],
    llm=deepseekllm
)

profile_analyzer = Agent(
    role='Profile Analysis Specialist',
    goal='Analyze candidate profiles to identify strengths, weaknesses, and unique selling points',
    backstory="You are a career coach with years of experience in helping professionals "
        "present their experience effectively. You excel at identifying transferable "
        "skills and unique value propositions.",
    verbose=True,
    allow_delegation=False,
    tools=[scrape_tool],
    llm=deepseekllm
)

matching_specialist = Agent(
    role='Career Matching Specialist',
    goal='Evaluate job-candidate fit and provide strategic application advice',
    backstory="You are a senior career advisor who specializes in helping candidates "
        "position themselves effectively for roles. You excel at identifying gaps "
        "and suggesting concrete steps for improvement.",
    verbose=True,
    allow_delegation=False,
    llm=deepseekllm
)

analyze_job = Task(
    description="Analyze the job posting at {job_url}."
        "Focus on:"
        "1. Key technical requirements"
        "2. Soft skills and cultural fit indicators"
        "3. Implicit requirements and nice-to-haves"
        "4. Company values and culture signals"
        "Provide a structured analysis that can be used by the matching specialist.",
    expected_output="Structured analysis of job requirements including necessary "
        "skills, qualifications experiences and company culture.",
    agent=job_analyzer
)

analyze_profile = Task(
    description="Analyze the candidate profile at {profile_url}."
        "Focus on:"
        "1. Technical skills and experience"
        "2. Demonstrated soft skills"
        "3. Career progression and achievements"
        "4. Unique selling points"
        "Provide a structured analysis that can be used by the matching specialist.",
    expected_output="Structured analysis of candidate profile including key skills, experiences, contributions, interests, and "
        "communication style.", 
    agent=profile_analyzer
)

provide_matching_advice = Task(
    description="Using the analyses from both the job and profile specialists:"
        "1. Evaluate the overall match percentage"
        "2. Identify key strengths to emphasize"
        "3. Point out gaps that need addressing"
        "4. Provide specific recommendations for:"
        "- Resume adjustments"
        "- Cover letter points"
        "- Interview preparation"
        "- Skill development priorities"
        "Provide actionable advice that the candidate can implement immediately.",
    expected_output="Detailed matching advice including strengths, gaps, and "
        "recommendations for resume, cover letter, interview preparation, and skill development.",
    context=[analyze_job, analyze_profile],
    agent=matching_specialist
)

application_crew = Crew(
    agents=[job_analyzer, profile_analyzer, matching_specialist],
    tasks=[analyze_job, analyze_profile, provide_matching_advice],
    verbose=True
)

result = application_crew.kickoff(
        inputs={
            "job_url": "https://www.linkedin.com/jobs/view/4136748557",
            "profile_url": "https://www.linkedin.com/in/serkandogantekin"
        }
    )