from typing import Dict, List
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import BaseMessage
from retrievers import tavily_tool, duckduckgo_tool
from langchain.prompts import PromptTemplate
from models.state import ResearchResult, ResearchNode


class Researcher:
    def __init__(self, llm, tools):
        """
        Initialize Researcher with LLM and tools.

        Args:
            llm: The LLM to use for research
        """
        self.llm = llm
        self.tools = tools

        from langchain import hub

        self.prompt = hub.pull("hwchase17/react")
        # prompt = """
        # You are a helpful assistant that can search the web for information and gather as much information as possible.
        # Answer the following questions as best you can. You have access to the following tools:

        # {tools}

        # Use the following format:

        # Question: the input question you must answer
        # Thought: you should always think about what to do
        # Action: the action to take, should be one of [{tool_names}]
        # Action Input: the input to the action
        # Observation: the result of the action
        # ... (this Thought/Action/Action Input/Observation can repeat N times)
        # Thought: I now know the final answer
        # Final Answer: the final answer to the original input question

        # Begin!

        # Question: {input}
        # Thought:{agent_scratchpad}
        # """
        # self.prompt = PromptTemplate(template=prompt)

        self.agent = create_react_agent(llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=10,
            return_intermediate_steps=True,  # Ensure we get all intermediate steps
        )

    def research(self, state: Dict) -> Dict:
        """
        Process messages in the state using the research agent.

        Args:
            state: The current state containing messages and other data

        Returns:
            Dict containing both the processed messages, structured output and raw search results

        Raises:
            ValueError: If messages are invalid
            Exception: If research processing fails
        """
        try:
            messages = state.get("messages", [])
            if not messages:
                raise ValueError("No messages found in state")

            # Get the last message content for research
            last_message = (
                messages[-1].content
                if hasattr(messages[-1], "content")
                else messages[-1]
            )

            # Execute research
            response = self.agent_executor.invoke(
                {"input": last_message, "agent_scratchpad": ""}
            )

            # Extract and structure search results
            raw_search_results = []

            for step in response.get("intermediate_steps", []):
                action = step[0]  # The action taken
                result = step[1]  # The result of the action

                if isinstance(result, str) and any(
                    tool.name in action.tool for tool in self.tools
                ):
                    # Store raw search result
                    raw_result = {
                        "tool": action.tool,
                        "query": action.tool_input,
                        "result": result,
                    }
                    raw_search_results.append(raw_result)

            # Create a comprehensive message that includes all data
            research_message = {
                "summary": response["output"],
            }
            if len(raw_search_results) > 0:
                research_message.update({"raw_search_results": raw_search_results})

            return {
                "messages": [str(research_message)],  # Include all data in messages
            }

        except Exception as e:
            raise Exception(f"Error during research: {str(e)}")
