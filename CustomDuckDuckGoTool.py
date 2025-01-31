from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun

class CustomDuckDuckGoTool(BaseTool):
    name: str = "Search the internet with DuckDuckGo"
    description: str = "A tool that can be used to search the internet with a search query."

    def _run(self, query: str) -> str:
        # Ensure the DuckDuckGoSearchRun is invoked properly.
        duckduckgo_tool = DuckDuckGoSearchRun()
        response = duckduckgo_tool.invoke(query)
        return response

    def _get_tool(self):
        # Create an instance of the tool when needed
        return CustomDuckDuckGoTool()