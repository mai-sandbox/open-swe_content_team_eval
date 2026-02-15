#!/usr/bin/env python3
"""Test script to verify all imports are working correctly."""

try:
    from typing import List, Annotated, TypedDict, Literal
    print("✅ typing imports successful")
    
    from dotenv import load_dotenv
    print("✅ dotenv import successful")
    
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    print("✅ langchain_core.messages imports successful")
    
    from langchain_anthropic import ChatAnthropic
    print("✅ langchain_anthropic import successful")
    
    from langchain_core.tools import tool
    print("✅ langchain_core.tools import successful")
    
    from langgraph.graph import StateGraph, START, END
    print("✅ langgraph.graph imports successful")
    
    from langgraph.graph.message import add_messages
    print("✅ langgraph.graph.message import successful")
    
    from langgraph.types import Send
    print("✅ langgraph.types import successful")
    
    print("\n🎉 All imports are working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
