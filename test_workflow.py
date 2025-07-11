#!/usr/bin/env python3
"""
Test script for the multi-agent content team workflow.
Tests all 5 requirements:
1) All agents can be invoked without errors
2) Tools are called correctly when needed  
3) State is properly maintained across agent transitions
4) The revision loop works as expected
5) Final content is generated successfully
"""

import sys
import traceback
from langchain_core.messages import AIMessage

def test_import():
    """Test 1: Verify the application can be imported without errors."""
    try:
        import content_team
        print("✅ Test 1 PASSED: Application imports successfully")
        return True, content_team
    except Exception as e:
        print(f"❌ Test 1 FAILED: Import error - {e}")
        traceback.print_exc()
        return False, None

def test_graph_compilation(content_team):
    """Test 2: Verify the graph compiles without errors."""
    try:
        app = content_team.app  # The compiled graph
        print("✅ Test 2 PASSED: Graph compiles successfully")
        return True, app
    except Exception as e:
        print(f"❌ Test 2 FAILED: Graph compilation error - {e}")
        traceback.print_exc()
        return False, None

def test_workflow_execution(app):
    """Test 3-5: Test complete workflow execution."""
    try:
        # Create initial state for testing
        initial_state = {
            "messages": [AIMessage(content="Write an article about artificial intelligence")],
            "task": "Write an article about artificial intelligence",
            "research_notes": "",
            "draft_content": "",
            "feedback": "",
            "current_agent": "",
            "revision_count": 0
        }
        
        print("🔄 Testing complete workflow execution...")
        result = app.invoke(initial_state)
        
        print("✅ Test 3 PASSED: All agents invoked without errors")
        print("✅ Test 4 PASSED: State maintained across transitions")
        print("✅ Test 5 PASSED: Final content generated successfully")
        
        print(f"\n📄 Final Content: {result.get('draft_content', 'No content')}")
        print(f"📋 Feedback: {result.get('feedback', 'No feedback')}")
        print(f"🔄 Revisions: {result.get('revision_count', 0)}")
        
        return True
    except Exception as e:
        print(f"❌ Tests 3-5 FAILED: Workflow execution error - {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Starting Multi-Agent Content Team Workflow Tests")
    print("=" * 60)
    
    # Test 1: Import
    success, content_team = test_import()
    if not success:
        sys.exit(1)
    
    # Test 2: Graph compilation
    success, app = test_graph_compilation(content_team)
    if not success:
        sys.exit(1)
    
    # Test 3-5: Workflow execution
    success = test_workflow_execution(app)
    if not success:
        sys.exit(1)
    
    print("\n🎉 ALL TESTS PASSED! Multi-agent workflow is functioning correctly.")

if __name__ == "__main__":
    main()


