import textwrap
import os
from camel.agents.chat_agent import ChatAgent
from camel.messages.base import BaseMessage
from camel.models.model_factory import ModelFactory
from camel.tasks.task import Task
from camel.toolkits.function_tool import FunctionTool
from camel.toolkits.search_toolkit import SearchToolkit
from camel.types.enums import ModelPlatformType
from camel.types.enums import ModelType
from camel.societies.workforce.workforce import Workforce
from dotenv import load_dotenv
from typing import Dict
from naptha_sdk.schemas import *
from naptha_sdk.user import sign_consumer_id
from naptha_sdk.utils import get_logger
from judge_agent.schemas import InputSchema

logger = get_logger(__name__)
load_dotenv()
def make_judge(
    persona: str,
    example_feedback: str,
    criteria: str,
) -> ChatAgent:
    msg_content = textwrap.dedent(
        f"""\
        You are a judge in a hackathon.
        This is your persona that you MUST act with: {persona}
        Here is an example feedback that you might give with your persona, you MUST try your best to align with this:
        {example_feedback}
        When evaluating projects, you must use the following criteria:
        {criteria}
        You also need to give scores based on these criteria, from 1-4. The score given should be like 3/4, 2/4, etc.
        """  # noqa: E501
    )

    sys_msg = BaseMessage.make_assistant_message(
        role_name="Hackathon Judge",
        content=msg_content,
    )

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
    )

    agent = ChatAgent(
        system_message=sys_msg,
        model=model,
    )

    return agent


def judge_agent():
    proj_content = textwrap.dedent(
        """\
        Project name: CAMEL-Powered Adaptive Learning Assistant
        How does your project address a real problem: Our CAMEL-Powered Adaptive Learning Assistant addresses the challenge of personalized education in an increasingly diverse and fast-paced learning environment. Traditional one-size-fits-all approaches to education often fail to meet the unique needs of individual learners, leading to gaps in understanding and reduced engagement. Our project leverages CAMEL-AI's advanced capabilities to create a highly adaptive, intelligent tutoring system that can understand and respond to each student's learning style, pace, and knowledge gaps in real-time.
        Explain your tech and which parts work: Our system utilizes CAMEL-AI's in-context learning and multi-domain application features to create a versatile learning assistant. The core components include:
        1. Learner Profile Analysis: Uses natural language processing to assess the student's current knowledge, learning preferences, and goals.
        2. Dynamic Content Generation: Leverages CAMEL-AI to create personalized learning materials, explanations, and practice questions tailored to each student's needs.
        3. Adaptive Feedback Loop: Continuously analyzes student responses and adjusts the difficulty and style of content in real-time.
        4. Multi-Modal Integration: Incorporates text, images, and interactive elements to cater to different learning styles.
        5. Progress Tracking: Provides detailed insights into the student's learning journey, identifying strengths and areas for improvement.
        Currently, we have successfully implemented the Learner Profile Analysis and Dynamic Content Generation modules. The Adaptive Feedback Loop is partially functional, while the Multi-Modal Integration and Progress Tracking features are still in development.
        """  # noqa: E501
    )

    # Create helper researcher agent with search tools.
    search_toolkit = SearchToolkit()
    search_tools = [
        FunctionTool(search_toolkit.search_google),
        FunctionTool(search_toolkit.search_duckduckgo),
    ]

    researcher_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
    )

    researcher_agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Researcher",
            content="You are a researcher who does research on AI and Open Sourced projects. You use web search to stay updated on the latest innovations and trends."
        ),
        model=researcher_model,
        tools=search_tools,
    )

    # Create venture capitalist judge
    vc_persona = f"""You are a venture capitalist who is obsessed with how projects can 
be scaled into "unicorn" companies. You peppers your speech with 
buzzwords like "disruptive," "synergistic," and "market penetration."
You do not concerned with technical details or innovation unless 
it directly impacts the business model."""

    vc_example_feedback = f"""Wow, this project is absolutely disruptive in the blockchain-enabled
        marketplace! I can definitely see synergistic applications in the
        FinTech ecosystem. The scalability is through the roof--this is
        revolutionary!"""

    vc_criteria = textwrap.dedent(
        """\
        ### **Applicability to Real-World Usage (1-4 points)**
        - **4**: The project directly addresses a significant real-world problem with a clear, scalable application.
        - **3**: The solution is relevant to real-world challenges but requires more refinement for practical or widespread use.
        - **2**: Some applicability to real-world issues, but the solution is not immediately practical or scalable.
        - **1**: Little or no relevance to real-world problems, requiring substantial changes for practical use.
        """  # noqa: E501
    )

    vc_agent = make_judge(
        vc_persona,
        vc_example_feedback,
        vc_criteria,
    )

    # Create experienced engineer judge
    eng_persona = f"""
You are an experienced engineer and a perfectionist. You are highly
detail-oriented and critical of any technical flaw, no matter how 
small. He evaluates every project as though it were going into a
mission-critical system tomorrow, so his feedback is thorough but 
often harsh."""

    eng_example_feedback = f"""There are serious code inefficiencies in this project. The
architecture is unstable, and the memory management is suboptimal.
I expect near-perfect performance, but this solution barely functions
under stress tests. It has potential, but it is nowhere near
deployment-ready."""

    eng_criteria = textwrap.dedent(
        """\
        ### **Technical Implementation (1-4 points)**
        - **4**: Flawless technical execution with sophisticated design, efficient performance, and robust architecture.
        - **3**: Strong technical implementation, though there may be areas for improvement or further development.
        - **2**: The project works, but technical limitations or inefficiencies hinder its overall performance.
        - **1**: Poor technical implementation with major issues in functionality, coding, or structure.
        """  # noqa: E501
    )

    eng_agent = make_judge(
        eng_persona,
        eng_example_feedback,
        eng_criteria,
    )

    # Create AI founder judge
    founder_persona = f"""You are a well-known AI startup founder who is always looking for the
"next big thing" in AI. You value bold, inventive ideas and
prioritizes projects that break new ground over those that improve 
existing systems."""

    founder_example_feedback = f"""This is interesting, but I have seen similar approaches before. I am 
looking for something that pushes boundaries and challenges norms. 
What is the most revolutionary part of this project? Let us see what 
is trending on Internet to make sure this is not already out there!"""

    founder_criteria = textwrap.dedent(
        """\
        ### **Innovation (1-4 points)**
        - **4**: The project showcases a groundbreaking concept or a unique approach that significantly departs from existing methods.
        - **3**: The project demonstrates a novel twist on known solutions or introduces some innovative aspects.
        - **2**: Some level of innovation is present, but the project largely builds on existing ideas without major new contributions.
        - **1**: Little or no innovation; the project is based on standard approaches with minimal creativity.
        """  # noqa: E501
    )

    founder_agent = make_judge(
        founder_persona,
        founder_example_feedback,
        founder_criteria,
    )

    # Create CAMEL contributor judge
    contributor_persona = f"""You are a contributor to the CAMEL-AI project and is always excited 
to see how people are using it. You are kind and optimistic, always 
offering positive feedback, even for projects that are still rough 
around the edges."""

    contributor_example_feedback = f"""Oh, I love how you have implemented CAMEL-AI here! The use of its
adaptive learning capabilities is fantastic, and you have really
leveraged the contextual reasoning in a great way! Let me just pull
up the GitHub README to check if there is any more potential 
optimizations."""

    contributor_criteria = textwrap.dedent(
        """\
        ### **Use of CAMEL-AI (1-4 points)**
        - **4**: Excellent integration of CAMEL-AI, fully leveraging its advanced features like in-context learning, adaptability, or multi-domain applications.
        - **3**: Good use of CAMEL-AI, but there are opportunities to exploit more of its advanced capabilities.
        - **2**: Limited use of CAMEL-AI, relying mostly on basic features without taking advantage of its full potential.
        - **1**: CAMEL-AI integration is minimal or poorly implemented, adding little value to the project.
        """  # noqa: E501
    )

    contributor_agent = make_judge(
        contributor_persona,
        contributor_example_feedback,
        contributor_criteria,
    )

    workforce = Workforce("Hackathon Judges")

    workforce.add_single_agent_worker(
        'Visionary Veronica (Judge), a venture capitalist who is '
        'obsessed with how projects can be scaled into "unicorn" companies',
        worker=vc_agent,
    ).add_single_agent_worker(
        'Critical John (Judge), an experienced engineer and a'
        ' perfectionist.',
        worker=eng_agent,
    ).add_single_agent_worker(
        'Innovator Iris (Judge), a well-known AI startup founder who'
        ' is always looking for the "next big thing" in AI.',
        worker=founder_agent,
    ).add_single_agent_worker(
        'Friendly Frankie (Judge), a contributor to the CAMEL-AI '
        'project and is always excited to see how people are using it.',
        worker=contributor_agent,
    ).add_single_agent_worker(
        'Researcher Rachel (Helper), a researcher who does online searches to'
        'find the latest innovations and trends on AI and Open Sourced '
        'projects.',
        worker=researcher_agent,
    )


    task = Task(content="Evaluate the hackathon project. First, do some research on "
    "the infomation related to the project, then each judge should give a"
    " score accordingly. Finally, list the opinions from each judge while"
    " preserving the judge's unique identity, along with the score and"
    " judge name, and also give a final summary of the opinions.",
        expected_output=module_run["inputs"].get("expected_output", "Analysis results"),
        agent=researcher_agent,
        human_input=False
    )

    task = workforce.process_task(task)

    return task.result

def run(module_run: Dict, *args, **kwargs):
    """
    Modified run function that creates and executes the agent.
    If 'func_name' is 'judge_agent', return the result from the agent function.
    """
    agent_run_input = AgentRunInput(**module_run)
    inputs_obj = InputSchema(**agent_run_input.inputs)
    func_to_call = globals().get(inputs_obj.func_name)
    
    if not func_to_call:
        raise ValueError(f"Function '{inputs_obj.func_name}' not found.")
    
    if inputs_obj.func_name == "judge_agent":
        return judge_agent()
    else:
        import inspect
        sig = inspect.signature(func_to_call)
        if len(sig.parameters) == 0:
            return func_to_call()
        else:
            tool_input_class = (
                globals().get(inputs_obj.input_type)
                if inputs_obj.input_type else None
            )
            input_data = (
                tool_input_class(**inputs_obj.func_input_data)
                if tool_input_class else inputs_obj.func_input_data
            )
            return func_to_call(input_data)

if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    import os

    naptha = Naptha()
    deployment = asyncio.run(
        setup_module_deployment(
            "agent",
            "judge_agent/configs/deployment.json",
            node_url=os.getenv("NODE_URL"),
            user_id=None,
            load_persona_data=False,
            is_subdeployment=False
        )
    )
    example_inputs = {
        "description": "Evaluate the CAMEL-Powered Adaptive Learning Assistant project",
        "expected_output": "A comprehensive evaluation of the project, including scores and feedback from multiple judges"
    }
    input_params = {
        "func_name": "judge_agent",
        "func_input_data": example_inputs
    }
    module_run = {
        "inputs": input_params,
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }
    response = run(module_run)
    print(response)