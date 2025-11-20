# Part of agent.py --> Follow https://google.github.io/adk-docs/get-started/quickstart/ to learn the setup

import asyncio
import os
import sys
from google.adk.agents import LoopAgent, LlmAgent, BaseAgent, SequentialAgent
from google.genai import types
from google.adk.runners import InMemoryRunner
from google.adk.agents.invocation_context import InvocationContext
from google.adk.tools.tool_context import ToolContext
from typing import AsyncGenerator, Optional
from google.adk.events import Event, EventActions
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# --- Constants ---
APP_NAME = "webapp_dev_app" # webapp Development App Name
USER_ID = "dev_user_01"
SESSION_ID_BASE = "dev_test_refine_session" # Development Session ID
GEMINI_MODEL = "gemini-2.5-flash"
STATE_REQUIREMENT = "requirement"
STATE_USER_REQUEST = "user_request"

# --- State Keys ---
STATE_CURRENT_PROJECT = "current_project"
STATE_TEST_RESULTS = "test_results"
STATE_ISSUES = "issues"
# Define the exact phrase the Tester should use to signal completion
COMPLETION_PHRASE = "All tests passed successfully."
TARGET_FOLDER_PATH = "/home/stream/AGENT"

dev_tools=[
    MCPToolset(
        connection_params=StdioConnectionParams(
            server_params = StdioServerParameters(
                command='desktop-commander', # 修正: 使用可执行文件名
                args=[], # 修正: 移除 npx 和 -y 参数
                env=os.environ.copy()
            ),
            timeout=14
        ),
        tool_filter=[
            'start_process',
            'interact_with_process',
            'read_process_output',
            'force_terminate',
            'list_sessions',
            'list_processes',
            'kill_process'
        ]
    ),
    MCPToolset(
        connection_params=StdioConnectionParams(
            server_params = StdioServerParameters(
                command='mcp-server-filesystem', # 修正: 使用可执行文件名
                args=[
                    # 修正: 移除 -y 参数
                    os.path.abspath(TARGET_FOLDER_PATH),
                ],
            ),
            timeout=7
        ),
        # Optional: Filter which tools from the MCP server are exposed
        # tool_filter=['list_directory', 'read_file']
    )
]

test_tools=[
    MCPToolset(
        connection_params=StdioConnectionParams(
            server_params = StdioServerParameters(
                command='desktop-commander', # 修正: 使用可执行文件名
                args=[], # 修正: 移除 npx 和 -y 参数
                env=os.environ.copy()
            ),
            timeout=15
        ),        tool_filter=[
            'start_process',
            'interact_with_process',
            'read_process_output',
            'force_terminate',
            'list_sessions',
            'list_processes',
            'kill_process'
        ]
    ),
    MCPToolset(
        connection_params=StdioConnectionParams(
            server_params = StdioServerParameters(
                command='chrome-devtools-mcp', # 修正: 使用可执行文件名
                args=[] # 修正: 移除 npx 和 -y 参数
            ),
            timeout=40
        ),
    )
]

refine_tools=[
    MCPToolset(
        connection_params=StdioConnectionParams(
            server_params = StdioServerParameters(
                command='mcp-server-filesystem', # 修正: 使用可执行文件名
                args=[
                    # 修正: 移除 -y 参数
                    os.path.abspath(TARGET_FOLDER_PATH),
                ],
            ),
            timeout=8
        ),
    ),
]

# --- Tool Definition ---
def exit_loop(tool_context: ToolContext):
  """Call this function ONLY when all tests pass, signaling the development process should end."""
  print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
  tool_context.actions.escalate = True
  # Return empty dict as tools should typically return JSON-serializable output
  return {}

def get_user_input(tool_context: ToolContext, prompt: str = "Please provide your feedback or next requirement: "):
    """Pauses the agent to wait for user input from the command line."""
    print(f"\n>>> [User Input Needed] {prompt}")
    user_in = input(">>> ")
    return {"user_input": user_in}

# --- Agent Definitions ---

# STEP 1: Initial Developer Agent (Runs ONCE at the beginning)

initial_developer_agent = LlmAgent(
    name="InitialDeveloperAgent",
    model=GEMINI_MODEL,
    include_contents='none',
    instruction=f"""You are a flutter Web application development tasked with creating initial project.
    1.First summary a project name,and create the new project using command: flutter create <project_name>
    2.Design the main functions of the software.
    3.Modify the *first version* of project within a new folder based on the requirement provided below.

    Requirement:
    {{requirement}}

    Output *only* the project absolute folder path and the project description.
""",
    description="Creates the initial project implementation based on the requirement.",
    tools=dev_tools,
    output_key=STATE_CURRENT_PROJECT
)

# STEP 2a: Tester Agent (Inside the Development Loop)

tester_agent_in_loop = LlmAgent(
    name="TesterAgent",
    model=GEMINI_MODEL,
    include_contents='none',
    instruction=f"""You are a Web application Tester reviewing project for functionality and quality.

    **Project to Test:**
    ```
    {{current_project}}
    ```

    **Initial Requirements:**
    {{requirement}}

    **Latest User Request:**
    {{user_request}}

    **Task:**
    1.Record all features to be tested and their completion status.
    2.Determine whether the core functionality of the software is complete. If all features have been tested and there are no new features to test, You MUST call the 'exit_loop' function.
    3.Plan a feature to be tested, prioritizing previously untested features or new features.
    4.Clearly define the testing process for this feature.
    5.Navigate to the project folder and Run the project using flutter run -d chrome --web-port 8080
    6.Test the functionality of http://localhost:8080/ using devtools according to the testing process.
    7.Record test_results. If a test fails, output an issues report for other agents to fix the problem.
    8.If all tests pass, output the "{COMPLETION_PHRASE}".

    Do some necessary explanations. Output only the test results/issues report OR the exact completion phrase.
""",
    description="Tests the current project, providing issues if problems are found, otherwise signals completion.",
    tools=[exit_loop]+test_tools,
    output_key=STATE_TEST_RESULTS
)


# STEP 2b: project Refiner/Exiter Agent (Inside the Development Loop)
project_refiner_agent_in_loop = LlmAgent(
    name="projectRefinerAgent",
    model=GEMINI_MODEL,
    # Relies solely on state via placeholders
    include_contents='none',
    instruction=f"""You are a Web application Developer refining project based on test results OR exiting the process.
    **Current project:**
    ```
    {{current_project}}
    ```
    **Test Results/Issues:**
    {{test_results}}

    **Task:**
    Analyze the 'Test Results/Issues'.Modify the project file based on the error report.
    IF the test results are *exactly* "{COMPLETION_PHRASE}":

    Do not add explanations. Either output the project project path
""",
    description="Refines the project based on test results, or calls exit_loop if tests indicate completion.",
    tools=refine_tools, # Provide the exit_loop tool
    output_key=STATE_CURRENT_PROJECT # Overwrites state['current_project'] with the refined version
)


# STEP 2: Development Loop Agent (Test -> Refine cycle)
development_loop = LoopAgent(
    name="DevelopmentLoop",
    # Agent order is crucial: Test first, then Refine/Exit
    sub_agents=[
        tester_agent_in_loop,
        project_refiner_agent_in_loop,
    ],
    max_iterations=20 # Limit loops
)

# STEP 3: Interactive Session Agents (Continuous Improvement)

user_interface_agent = LlmAgent(
    name="UserInterfaceAgent",
    model=GEMINI_MODEL,
    include_contents='none',
    instruction=f"""You are the interface between the user and the development process.
    1. Ask the user what they want to do next using the `get_user_input` tool.
    2. If the user wants to quit (e.g., types 'exit', 'quit', 'done'), call the `exit_loop` tool to end the session.
    3. Otherwise, output the user's request exactly.

    Output *only* the user's request.
""",
    description="Gets instructions from the user.",
    tools=[get_user_input, exit_loop],
    output_key=STATE_USER_REQUEST
)

feature_developer_agent = LlmAgent(
    name="FeatureDeveloperAgent",
    model=GEMINI_MODEL,
    include_contents='none',
    instruction=f"""You are a Flutter Web Developer.
    **Current Project:**
    ```
    {{current_project}}
    ```
    **User Request:**
    ```
    {{user_request}}
    ```

    **Task:**
    Implement the user's request by modifying the project files or running commands.
    Use the available tools to explore the code and apply changes.

    Output the project path.
""",
    description="Implements new features based on user request.",
    tools=dev_tools, # Needs filesystem and process control
    output_key=STATE_CURRENT_PROJECT
)

interactive_session = LoopAgent(
    name="InteractiveSession",
    sub_agents=[
        user_interface_agent,
        feature_developer_agent,
        development_loop # Re-verify the project after features are added
    ],
    max_iterations=100
)

# STEP 4: Overall Sequential Pipeline
# For ADK tools compatibility, the root agent must be named `root_agent`
root_agent = SequentialAgent(
    name="WebApplicationDevelopmentPipeline",
    sub_agents=[
        initial_developer_agent, # Run first to create initial project
        development_loop,        # Stabilize initial project
        interactive_session      # Enter interactive mode for continuous improvement
    ],
    description="Creates initial project and then iteratively tests and refines it, allowing for user-driven continuous improvement."
)
