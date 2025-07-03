"""
Multi-agent content creation team using LangGraph.
"""

from typing import Annotated, TypedDict, Literal, List
from dotenv import load_dotenv
import logging

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Shared state for the team
class TeamState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    task: str
    research_notes: str
    draft_content: str
    feedback: str
    current_agent: str
    revision_count: int

# State validation functions
def validate_state(state: TeamState) -> bool:
    """Validate that the state contains required fields."""
    try:
        required_fields = ['messages', 'task', 'research_notes', 'draft_content', 
                          'feedback', 'current_agent', 'revision_count']
        
        for field in required_fields:
            if field not in state:
                logging.error(f"Missing required state field: {field}")
                return False
        
        # Validate field types
        if not isinstance(state['messages'], list):
            logging.error("State field 'messages' must be a list")
            return False
            
        if not isinstance(state['task'], str):
            logging.error("State field 'task' must be a string")
            return False
            
        if not isinstance(state['revision_count'], int):
            logging.error("State field 'revision_count' must be an integer")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Error validating state: {e}")
        return False

def safe_get_state_field(state: TeamState, field: str, default=""):
    """Safely get a state field with a default value."""
    try:
        return state.get(field, default)
    except Exception as e:
        logging.error(f"Error accessing state field '{field}': {e}")
        return default

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
    """Create research agent with error handling."""
    try:
        model = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.1)
        return model.bind_tool([web_research])
    except Exception as e:
        logging.error(f"Error creating research agent: {e}")
        raise

def create_writer_agent():
    """Create writer agent with error handling."""
    try:
        model = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.7)
        return model
    except Exception as e:
        logging.error(f"Error creating writer agent: {e}")
        raise

def create_reviewer_agent():
    """Create reviewer agent with error handling."""
    try:
        model = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.3)
        return model.bind_tool([fact_check])
    except Exception as e:
        logging.error(f"Error creating reviewer agent: {e}")
        raise

# Agent nodes with comprehensive error handling
def research_agent_node(state: TeamState):
    """Research agent gathers information with error handling."""
    try:
        # Validate state
        if not validate_state(state):
            logging.error("Invalid state in research_agent_node")
            return {
                "messages": state.get("messages", []) + [AIMessage(content="Error: Invalid state for research agent")],
                "current_agent": "researcher",
                "research_notes": "Error: Could not complete research due to invalid state"
            }
        
        model = create_research_agent()
        task = safe_get_state_field(state, 'task', 'No task specified')
        
        system_msg = SystemMessage(content=f"""
        You are a research agent. Your task is to research: {task}
        
        Use the web_research tool to gather information.
        Then pass your findings to the writer agent.
        """)
        
        messages = [system_msg] + state["messages"]
        
        # Invoke model with error handling
        try:
            response = model.invoke(messages)
        except Exception as e:
            logging.error(f"Error invoking research model: {e}")
            error_response = AIMessage(content=f"Research failed: {str(e)}")
            return {
                "messages": state["messages"] + [error_response],
                "current_agent": "researcher",
                "research_notes": f"Error during research: {str(e)}"
            }
        
        # Handle tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            try:
                tool_messages = []
                for tool_call in response.tool_calls:
                    if tool_call['name'] == 'web_research':
                        try:
                            result = web_research.invoke(tool_call['args'])
                            tool_messages.append(ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call['id']
                            ))
                        except Exception as e:
                            logging.error(f"Error executing web_research tool: {e}")
                            tool_messages.append(ToolMessage(
                                content=f"Tool execution failed: {str(e)}",
                                tool_call_id=tool_call['id']
                            ))
                
                # Get final response after tool execution
                try:
                    final_response = model.invoke(messages + [response] + tool_messages)
                except Exception as e:
                    logging.error(f"Error getting final response from research model: {e}")
                    final_response = AIMessage(content=f"Research completed with errors: {str(e)}")
                
                return {
                    "messages": state["messages"] + [response] + tool_messages + [final_response],
                    "current_agent": "researcher",
                    "research_notes": final_response.content
                }
            except Exception as e:
                logging.error(f"Error handling tool calls in research agent: {e}")
                return {
                    "messages": state["messages"] + [response],
                    "current_agent": "researcher", 
                    "research_notes": f"Research completed with tool errors: {str(e)}"
                }
        else:
            return {
                "messages": state["messages"] + [response],
                "current_agent": "researcher",
                "research_notes": response.content
            }
            
    except Exception as e:
        logging.error(f"Unexpected error in research_agent_node: {e}")
        return {
            "messages": state.get("messages", []) + [AIMessage(content=f"Research agent failed: {str(e)}")],
            "current_agent": "researcher",
            "research_notes": f"Critical error: {str(e)}"
        }

def writer_agent_node(state: TeamState):
    """Writer agent creates content with error handling."""
    try:
        # Validate state
        if not validate_state(state):
            logging.error("Invalid state in writer_agent_node")
            return {
                "messages": state.get("messages", []) + [AIMessage(content="Error: Invalid state for writer agent")],
                "current_agent": "writer",
                "draft_content": "Error: Could not create content due to invalid state"
            }
        
        model = create_writer_agent()
        task = safe_get_state_field(state, 'task', 'No task specified')
        research_notes = safe_get_state_field(state, 'research_notes', 'No research available')
        
        system_msg = SystemMessage(content=f"""
        You are a creative writer. Create engaging content based on:
        Task: {task}
        Research: {research_notes}
        
        Write clear, engaging content that incorporates the research findings.
        """)
        
        messages = [system_msg] + state["messages"]
        
        # Invoke model with error handling
        try:
            response = model.invoke(messages)
        except Exception as e:
            logging.error(f"Error invoking writer model: {e}")
            error_response = AIMessage(content=f"Content creation failed: {str(e)}")
            return {
                "messages": state["messages"] + [error_response],
                "current_agent": "writer",
                "draft_content": f"Error during content creation: {str(e)}"
            }
        
        return {
            "messages": state["messages"] + [response],
            "current_agent": "writer",
            "draft_content": response.content
        }
        
    except Exception as e:
        logging.error(f"Unexpected error in writer_agent_node: {e}")
        return {
            "messages": state.get("messages", []) + [AIMessage(content=f"Writer agent failed: {str(e)}")],
            "current_agent": "writer",
            "draft_content": f"Critical error: {str(e)}"
        }
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
    if response.tool_calls:
        # Execute tool calls
        tool_messages = []
        for tool_call in response.tool_calls:
            if tool_call["name"] == "web_research":
                tool_result = web_research.invoke(tool_call["args"])
                tool_messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                ))
        
        # Get final response after tool execution
        if tool_messages:
            final_response = model.invoke(messages + [response] + tool_messages)
            research_notes = final_response.content
        else:
            research_notes = response.content
    else:
        research_notes = response.content
    
    return {
        "messages": state["messages"] + [response],
        "research_notes": research_notes,
        "current_agent": "researcher"
    }

def writer_agent_node(state: TeamState):
    """Writer agent creates content based on research."""
    model = create_writer_agent()
    
    system_msg = SystemMessage(content=f"""
    You are a writer agent. Create engaging content about: {state['task']}
    
    Use this research: {state['research_notes']}
    
    Write a comprehensive article and then hand off to the reviewer.
    """)
    
    messages = [system_msg] + state["messages"]
    response = model.invoke(messages)
    
    return {
        "messages": state["messages"] + [response],
        "draft_content": response.content,
        "current_agent": "writer"
    }

def reviewer_agent_node(state: TeamState):
    """Reviewer agent provides feedback with error handling."""
    try:
        # Validate state
        if not validate_state(state):
            logging.error("Invalid state in reviewer_agent_node")
            return {
                "messages": state.get("messages", []) + [AIMessage(content="Error: Invalid state for reviewer agent")],
                "current_agent": "reviewer",
                "feedback": "Error: Could not review content due to invalid state",
                "revision_count": state.get("revision_count", 0) + 1
            }
        
        model = create_reviewer_agent()
        task = safe_get_state_field(state, 'task', 'No task specified')
        draft_content = safe_get_state_field(state, 'draft_content', 'No content to review')
        revision_count = state.get('revision_count', 0)
        
        system_msg = SystemMessage(content=f"""
        You are a content reviewer. Review this content for:
        Task: {task}
        Content: {draft_content}
        
        Use the fact_check tool to verify information.
        Provide constructive feedback. If content needs revision, be specific about improvements needed.
        Current revision count: {revision_count}
        """)
        
        messages = [system_msg] + state["messages"]
        
        # Invoke model with error handling
        try:
            response = model.invoke(messages)
        except Exception as e:
            logging.error(f"Error invoking reviewer model: {e}")
            error_response = AIMessage(content=f"Review failed: {str(e)}")
            return {
                "messages": state["messages"] + [error_response],
                "current_agent": "reviewer",
                "feedback": f"Error during review: {str(e)}",
                "revision_count": state.get("revision_count", 0) + 1
            }
        
        # Handle tool calls if present
        feedback_content = response.content
        if hasattr(response, 'tool_calls') and response.tool_calls:
            try:
                tool_messages = []
                for tool_call in response.tool_calls:
                    if tool_call['name'] == 'fact_check':
                        try:
                            result = fact_check.invoke(tool_call['args'])
                            tool_messages.append(ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call['id']
                            ))
                        except Exception as e:
                            logging.error(f"Error executing fact_check tool: {e}")
                            tool_messages.append(ToolMessage(
                                content=f"Fact-check failed: {str(e)}",
                                tool_call_id=tool_call['id']
                            ))
                
                # Get final response after tool execution
                try:
                    if tool_messages:
                        final_response = model.invoke(messages + [response] + tool_messages)
                        feedback_content = final_response.content
                except Exception as e:
                    logging.error(f"Error getting final response from reviewer model: {e}")
                    feedback_content = f"Review completed with errors: {str(e)}"
                
            except Exception as e:
                logging.error(f"Error handling tool calls in reviewer agent: {e}")
                feedback_content = f"Review completed with tool errors: {str(e)}"
        
        return {
            "messages": state["messages"] + [response],
            "feedback": feedback_content,
            "current_agent": "reviewer",
            "revision_count": state.get("revision_count", 0) + 1
        }
            
    except Exception as e:
        logging.error(f"Unexpected error in reviewer_agent_node: {e}")
        return {
            "messages": state.get("messages", []) + [AIMessage(content=f"Reviewer agent failed: {str(e)}")],
            "current_agent": "reviewer",
            "feedback": f"Critical error: {str(e)}",
            "revision_count": state.get("revision_count", 0) + 1
        }

# Routing logic
def route_from_researcher(state: TeamState) -> Literal["writer"]:
    """Route from researcher to writer."""
    return "writer"

def route_from_writer(state: TeamState) -> Literal["reviewer"]:
    """Route from writer to reviewer."""
    return "reviewer"

def route_from_reviewer(state: TeamState) -> Literal["writer_revision", "end"]:
    """Route from reviewer based on feedback."""
    feedback = state.get("feedback", "").lower()
    revision_count = state.get("revision_count", 0)
    
    if "revision" in feedback and revision_count < 2:
        return "writer_revision"
    else:
        return "end"

def route_from_writer_revision(state: TeamState) -> Literal["reviewer"]:
    """Route from writer revision back to reviewer."""
    return "reviewer"

# Legacy function for backward compatibility (not used in new routing)
def route_to_next_agent(state: TeamState) -> Literal["writer", "reviewer", "writer_revision", "end"]:
    """Legacy routing function - replaced by specific routing functions."""
    current = state.get("current_agent", "")
    
    if current == "researcher":
        return "writer"
    elif current == "writer":
        return "reviewer"  
    elif current == "reviewer":
        feedback = state.get("feedback", "").lower()
        revision_count = state.get("revision_count", 0)
        if "revision" in feedback and revision_count < 2:
            return "writer_revision"
        else:
            return "end"
    
    return "end"

def writer_revision_node(state: TeamState):
    """Writer revises content based on feedback with error handling."""
    try:
        # Validate state
        if not validate_state(state):
            logging.error("Invalid state in writer_revision_node")
            return {
                "messages": state.get("messages", []) + [AIMessage(content="Error: Invalid state for writer revision")],
                "current_agent": "writer",
                "draft_content": "Error: Could not revise content due to invalid state"
            }
        
        model = create_writer_agent()
        feedback = safe_get_state_field(state, 'feedback', 'No feedback provided')
        draft_content = safe_get_state_field(state, 'draft_content', 'No content to revise')
        research_notes = safe_get_state_field(state, 'research_notes', 'No research available')
        
        system_msg = SystemMessage(content=f"""
        Revise your content based on this feedback: {feedback}
        
        Original content: {draft_content}
        Research notes: {research_notes}
        
        Provide an improved version that addresses the feedback.
        """)
        
        messages = [system_msg] + state["messages"]
        
        # Invoke model with error handling
        try:
            response = model.invoke(messages)
        except Exception as e:
            logging.error(f"Error invoking writer revision model: {e}")
            error_response = AIMessage(content=f"Content revision failed: {str(e)}")
            return {
                "messages": state["messages"] + [error_response],
                "current_agent": "writer",
                "draft_content": f"Error during revision: {str(e)}"
            }
        
        return {
            "messages": state["messages"] + [response],
            "draft_content": response.content,
            "current_agent": "writer"
        }
        
    except Exception as e:
        logging.error(f"Unexpected error in writer_revision_node: {e}")
        return {
            "messages": state.get("messages", []) + [AIMessage(content=f"Writer revision failed: {str(e)}")],
            "current_agent": "writer",
            "draft_content": f"Critical error: {str(e)}"
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
    route_from_researcher,
    {
        "writer": "writer"
    }
)

graph_builder.add_conditional_edge(
    "writer", 
    route_from_writer,
    {
        "reviewer": "reviewer"
    }
)

graph_builder.add_conditional_edge(
    "reviewer",
    route_from_reviewer,
    {
        "writer_revision": "writer_revision", 
        "end": END
    }
)

graph_builder.add_conditional_edge(
    "writer_revision",
    route_from_writer_revision,
    {
        "reviewer": "reviewer"
    }
)

app = graph_builder.compile()

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













