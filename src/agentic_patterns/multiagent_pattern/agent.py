from textwrap import dedent

from agentic_patterns.multiagent_pattern.crew import Crew
from agentic_patterns.planning_pattern.react_agent import ReactAgent
from agentic_patterns.tool_pattern.tool import Tool


class Agent:
    """
    Represents an AI agent that can work as part of a team to complete tasks.

    This class implements an agent with dependencies, context handling, and task execution capabilities.
    It can be used in a multi-agent system where agents collaborate to solve complex problems.

    Attributes:
        name (str): The name of the agent.
        backstory (str): The backstory or background of the agent.
        task_description (str): A description of the task assigned to the agent.
        task_expected_output (str): The expected format or content of the task output.
        react_agent (ReactAgent): An instance of ReactAgent used for generating responses.
        dependencies (list[Agent]): A list of Agent instances that this agent depends on.
        dependents (list[Agent]): A list of Agent instances that depend on this agent.
        context (str): Accumulated context information from other agents.

    Args:
        name (str): The name of the agent.
        backstory (str): The backstory or background of the agent.
        task_description (str): A description of the task assigned to the agent.
        task_expected_output (str, optional): The expected format or content of the task output. Defaults to "".
        tools (list[Tool] | None, optional): A list of Tool instances available to the agent. Defaults to None.
        llm (str, optional): The name of the language model to use. Defaults to "llama-3.3-70b-versatile".
    """

    def __init__(
        self,
        name: str,
        backstory: str,
        task_description: str,
        task_expected_output: str = "",
        tools: list[Tool] | None = None,
        llm: str = "llama-3.3-70b-versatile",
    ):
        self.name = name
        self.backstory = backstory
        self.task_description = task_description
        self.task_expected_output = task_expected_output
        self.react_agent = ReactAgent(
            model=llm, system_prompt=self.backstory, tools=tools or []
        )

        self.dependencies: list[Agent] = []  # Agents that this agent depends on
        self.dependents: list[Agent] = []  # Agents that depend on this agent

        self.context = ""

        # Automatically register this agent to the active Crew context if one exists
        Crew.register_agent(self)

    def __repr__(self):
        return f"{self.name}"

    def __rshift__(self, other):
        """
        Defines the '>>' operator. This operator is used to indicate agent dependency.

        Args:
            other (Agent): The agent that depends on this agent.
        """
        self.add_dependent(other)
        return other  # Allow chaining

    def __lshift__(self, other):
        """
        Defines the '<<' operator to indicate agent dependency in reverse.

        Args:
            other (Agent): The agent that this agent depends on.

        Returns:
            Agent: The `other` agent to allow for chaining.
        """
        self.add_dependency(other)
        return other  # Allow chaining

    def __rrshift__(self, other):
        """
        Defines the '<<' operator.This operator is used to indicate agent dependency.

        Args:
            other (Agent): The agent that this agent depends on.
        """
        self.add_dependency(other)
        return self  # Allow chaining

    def __rlshift__(self, other):
        """
        Defines the '<<' operator when evaluated from right to left.
        This operator is used to indicate agent dependency in the normal order.

        Args:
            other (Agent): The agent that depends on this agent.

        Returns:
            Agent: The current agent (self) to allow for chaining.
        """
        self.add_dependent(other)
        return self  # Allow chaining

    def add_dependency(self, other):
        """
        Adds a dependency to this agent.

        Args:
            other (Agent | list[Agent]): The agent(s) that this agent depends on.

        Raises:
            TypeError: If the dependency is not an Agent or a list of Agents.
        """
        if isinstance(other, Agent):
            self.dependencies.append(other)
            other.dependents.append(self)
        elif isinstance(other, list) and all(isinstance(item, Agent) for item in other):
            for item in other:
                self.dependencies.append(item)
                item.dependents.append(self)
        else:
            raise TypeError("The dependency must be an instance or list of Agent.")

    def add_dependent(self, other):
        """
        Adds a dependent to this agent.

        Args:
            other (Agent | list[Agent]): The agent(s) that depend on this agent.

        Raises:
            TypeError: If the dependent is not an Agent or a list of Agents.
        """
        if isinstance(other, Agent):
            other.dependencies.append(self)
            self.dependents.append(other)
        elif isinstance(other, list) and all(isinstance(item, Agent) for item in other):
            for item in other:
                item.dependencies.append(self)
                self.dependents.append(item)
        else:
            raise TypeError("The dependent must be an instance or list of Agent.")

    def receive_context(self, input_data):
        """
        Receives and stores context information from other agents.

        Args:
            input_data (str): The context information to be added.
        """
        self.context += f"{self.name} received context: \n{input_data}"

    def create_prompt(self):
        """
        Creates a prompt for the agent based on its task description, expected output, and context.

        Returns:
            str: The formatted prompt string.
        """
        prompt = dedent(
            f"""
        You are an AI agent. You are part of a team of agents working together to complete a task.
        I'm going to give you the task description enclosed in <task_description></task_description> tags. I'll also give
        you the available context from the other agents in <context></context> tags. If the context
        is not available, the <context></context> tags will be empty. You'll also receive the task
        expected output enclosed in <task_expected_output></task_expected_output> tags. With all this information
        you need to create the best possible response, always respecting the format as describe in
        <task_expected_output></task_expected_output> tags. If expected output is not available, just create
        a meaningful response to complete the task.

        <task_description>
        {self.task_description}
        </task_description>

        <task_expected_output>
        {self.task_expected_output}
        </task_expected_output>

        <context>
        {self.context}
        </context>

        Your response:
        """
        ).strip()

        return prompt

    def run(self):
        """
        Runs the agent's task and generates the output.

        This method creates a prompt, runs it through the ReactAgent, and passes the output to all dependent agents.

        Returns:
            str: The output generated by the agent.
        """
        msg = self.create_prompt()
        output = self.react_agent.run(user_msg=msg)

        # Pass the output to all dependents
        for dependent in self.dependents:
            dependent.receive_context(output)
        return output
