"""
Multi-agent content creation team using LangGraph.
"""

from typing import Annotated, TypedDict, Literal, List
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send

load_dotenv()

# Shared state for the team
class TeamState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    task: str
    research_notes: str
    draft_content: str
    feedback: str
    current_agent: str
    revision_count: int

@tool
def web_research(topic: str) -> str:
    """Research information about a topic."""
    research_data = {
        "ai": "Artificial Intelligence is transforming industries through machine learning and automation...",
        "climate": "Climate change requires urgent action with renewable energy and carbon reduction...",
        "technology": "Latest tech trends include quantum computing, edge computing, and 5G networks..."
    }
    
    for keyword in research_data:
        if keyword in topic.lower():
            return f"Research on {topic}: {research_data[keyword]}"
    
    return f"General research findings for: {topic}"

@tool  
def fact_check(content: str) -> str:
    """Fact-check content for accuracy."""
    if len(content) < 50:
        return "Content too short to verify - needs more detail"
    return "Fact-check complete: Content appears accurate based on available information"

# Initialize agents
def create_research_agent():
    model = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.1)
    return model.bind_tool([web_research])

def create_writer_agent():
    model = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.7)
    return model

def create_reviewer_agent():
    model = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.3)
    return model.bind_tool([fact_check])

# Agent nodes
def research_agent_node(state: TeamState):
    """Research agent gathers information."""
    model = create_research_agent()
    
    system_msg = SystemMessage(content=f"""
    You are a research agent. Your task is to research: {state['task']}
    
    Use the web_research tool to gather information.
    Then pass your findings to the writer agent.
    """)
    
    messages = [system_msg] + state["messages"]
    response = model.invoke(messages)
    
    # Handle tool calls if present
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_results = []
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'web_research':
                # Invoke the web_research tool
                result = web_research(tool_call['args']['topic'])
                tool_results.append(result)
        
        # Create a follow-up message with tool results
        if tool_results:
            tool_message = AIMessage(content=f"Tool results: {'; '.join(tool_results)}")
            messages.append(response)
            messages.append(tool_message)
            # Get final response after tool execution
            final_response = model.invoke(messages)
            research_notes = '; '.join(tool_results)
        else:
            final_response = response
            research_notes = "Research completed - see message for details"
    else:
        final_response = response
        research_notes = "Research completed - see message for details"
    
    return {
        "messages": state["messages"] + [final_response],
        "research_notes": research_notes,
        "current_agent": "researcher",
        "task": state["task"],
        "draft_content": state.get("draft_content", ""),
        "feedback": state.get("feedback", ""),
        "revision_count": state.get("revision_count", 0)
    }

def writer_agent_node(state: TeamState):
    """Writer agent creates content based on research."""
    model = create_writer_agent()
    
    system_msg = SystemMessage(content=f"""
    You are a writer agent. Create engaging content about: {state['task']}
    
    Use this research: {state['research_notes']}
    
    Write a comprehensive article and then hand off to the reviewer.
    """)
    
    messages = [system_msg] + state["messages"]  # Preserve full message context
    response = model.invoke(messages)
    
    return {
        "messages": state["messages"] + [response],
        "draft_content": response.content,
        "current_agent": "writer",
        "task": state["task"],
        "research_notes": state.get("research_notes", ""),
        "feedback": state.get("feedback", ""),
        "revision_count": state.get("revision_count", 0)
    }

def reviewer_agent_node(state: TeamState):
    """Reviewer agent provides feedback."""
    model = create_reviewer_agent()
    
    system_msg = SystemMessage(content=f"""
    You are a reviewer agent. Review this content: {state['draft_content']}
    
    Use the fact_check tool to verify accuracy.
    Provide constructive feedback for improvement.
    
    If content needs major revision, send back to writer.
    If content is good, approve for publication.
    """)
    
    messages = [system_msg]
    response = model.invoke(messages)
    
    # Handle tool calls if present
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_results = []
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'fact_check':
                # Invoke the fact_check tool
                result = fact_check(tool_call['args']['content'])
                tool_results.append(result)
        
        # Create a follow-up message with tool results
        if tool_results:
            tool_message = AIMessage(content=f"Tool results: {'; '.join(tool_results)}")
            messages.append(response)
            messages.append(tool_message)
            # Get final response after tool execution
            final_response = model.invoke(messages)
            feedback_content = f"{final_response.content} (Tool results: {'; '.join(tool_results)})"
        else:
            final_response = response
            feedback_content = response.content
    else:
        final_response = response
        feedback_content = response.content
    
    return {
        "messages": [final_response],
        "feedback": feedback_content,
        "current_agent": "reviewer",
        "revision_count": state.get("revision_count", 0) + 1
    }

# Routing logic
def route_to_next_agent(state: TeamState) -> Literal["writer", "reviewer", "writer_revision", "end"]:
    """Route to next agent based on current state."""
    current = state.get("current_agent", "")
    
    if current == "researcher":
        return "writer"
    elif current == "writer":
        return "reviewer"  
    elif current == "reviewer":
        # Check if revision needed
        feedback = state.get("feedback", "").lower()
        revision_count = state.get("revision_count", 0)
        
        if "revision" in feedback and revision_count < 2:
            return "writer_revision"
        else:
            return "end"
    
    return "end"

def writer_revision_node(state: TeamState):
    """Writer revises content based on feedback."""
    model = create_writer_agent()
    
    system_msg = SystemMessage(content=f"""
    Revise your content based on this feedback: {state['feedback']}
    
    Original content: {state['draft_content']}
    Research notes: {state['research_notes']}
    
    Provide an improved version.
    """)
    
    response = model.invoke([system_msg])
    
    return {
        "messages": [response],
        "draft_content": response.content,
        "current_agent": "writer"
    }

graph_builder = StateGraph(TeamState)

# Add agent nodes
graph_builder.add_node("researcher", research_agent_node)
graph_builder.add_node("writer", writer_agent_node)
graph_builder.add_node("reviewer", reviewer_agent_node)
graph_builder.add_node("writer_revision", writer_revision_node)

graph_builder.add_edge(START, "researcher")

# Add conditional routing
graph_builder.add_conditional_edge(
    "researcher",
    route_to_next_agent,
    {
        "writer": "writer", 
        "end": END
    }
)

graph_builder.add_conditional_edge(
    "writer", 
    route_to_next_agent,
    {
        "reviewer": "reviewer", 
        "end": END
    }
)

graph_builder.add_conditional_edge(
    "reviewer",
    route_to_next_agent,
    {
        "writer_revision": "writer_revision", 
        "end": END
    }
)

app = graph_builder.build()

def main():
    """Run the multi-agent content team."""
    print("📝 Multi-Agent Content Creation Team Started!")
    print("Ask the team to create content on any topic.")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        user_task = input("\nContent request: ")
        if user_task.lower() == 'quit':
            break
        
        initial_state = {
            "messages": [AIMessage(content=user_task)],
            "task": user_task,
            "research_notes": "",
            "draft_content": "",
            "feedback": "",
            "current_agent": "",
            "revision_count": 0
        }
        
        try:
            print(f"\n🔄 Processing: {user_task}")
            
            result = app.invoke(initial_state)
            
            print(f"\n✅ Content Creation Complete!")
            print(f"📄 Final Content:\n{result.get('draft_content', 'No content generated')}")
            print(f"📋 Feedback: {result.get('feedback', 'No feedback')}")
            print(f"🔄 Revisions: {result.get('revision_count', 0)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()







