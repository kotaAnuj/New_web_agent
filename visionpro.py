import os
import time
import json
import pyautogui
import google.generativeai as genai
import torch
from PIL import Image
import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification
import threading
import queue
import logging
import streamlit as st
from typing import Dict, Any, List, Union

class DynamicComputerAssistant:
    def __init__(self, api_key: str):
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize AI models
        genai.configure(api_key=api_key)
        self.planner = genai.GenerativeModel('gemini-1.5-pro')
        self.executor = genai.GenerativeModel('gemini-1.5-pro')
        self.analyzer = genai.GenerativeModel('gemini-1.5-pro')
        
        # System components
        self.task_queue = queue.Queue()
        self.context_memory = []
        self.max_memory = 10
        
        # Initialize PyAutoGUI safely
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.5

    def safe_parse_json(self, text: str) -> Dict:
        """Safely parse JSON from AI response"""
        try:
            # Find JSON content between triple backticks if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            # Clean up common formatting issues
            text = text.replace("'", '"')
            text = text.replace('None', 'null')
            text = text.replace('True', 'true')
            text = text.replace('False', 'false')
            
            return json.loads(text)
        except Exception as e:
            self.logger.error(f"JSON parsing error: {e}")
            return {"error": f"Failed to parse response: {str(e)}"}

    def analyze_context(self, user_input: str) -> Dict:
        """Analyze user input and current context to understand the request"""
        context_prompt = """
        Analyze this user request and provide a JSON response with the following structure:
        {
            "intent": "user's primary goal",
            "required_access": ["list", "of", "required", "system", "access"],
            "risks": ["potential", "risks"],
            "resources": ["required", "resources"],
            "success_criteria": ["list", "of", "success", "criteria"]
        }
        
        User request: %s
        Recent context: %s
        """ % (user_input, str(self.context_memory[-3:] if self.context_memory else 'None'))
        
        try:
            response = self.analyzer.generate_content(context_prompt).text
            return self.safe_parse_json(response)
        except Exception as e:
            self.logger.error(f"Context analysis error: {e}")
            return {"error": str(e)}

    def create_action_plan(self, context_analysis: Dict) -> List[Dict]:
        """Create detailed plan of actions based on context analysis"""
        planning_prompt = """
        Create a detailed action plan as a JSON array of steps. Each step should have this structure:
        {
            "action": "specific_action",
            "params": {"param1": "value1"},
            "validation": "how to check if successful"
        }
        
        Context analysis: %s
        """ % str(context_analysis)
        
        try:
            response = self.planner.generate_content(planning_prompt).text
            return self.safe_parse_json(response)
        except Exception as e:
            self.logger.error(f"Planning error: {e}")
            return []

    def execute_action(self, action: Dict) -> bool:
        """Execute a single atomic action"""
        execution_prompt = """
        Generate Python code using PyAutoGUI to execute this action.
        Return ONLY executable Python code, no explanations.
        The code should be safe and include try/except blocks.
        
        Action: %s
        Screen state: %s
        """ % (str(action), str(self.get_screen_state()))
        
        try:
            response = self.executor.generate_content(execution_prompt).text
            
            # Extract code from response
            if "```python" in response:
                code = response.split("```python")[1].split("```")[0].strip()
            else:
                code = response.strip()
            
            # Create safe execution environment
            safe_globals = {
                'pyautogui': pyautogui,
                'time': time,
                'logging': logging
            }
            
            # Execute the code
            exec(code, safe_globals)
            time.sleep(0.5)
            return True
            
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            return False

    def get_screen_state(self) -> Dict:
        """Analyze current screen state"""
        try:
            screen_info = pyautogui.size()
            mouse_pos = pyautogui.position()
            active_window = None
            
            try:
                active_window = pyautogui.getActiveWindow()
            except:
                pass
                
            return {
                "screen_size": {"width": screen_info[0], "height": screen_info[1]},
                "mouse_position": {"x": mouse_pos[0], "y": mouse_pos[1]},
                "active_window": str(active_window) if active_window else None
            }
        except Exception as e:
            self.logger.error(f"Screen analysis error: {e}")
            return {}

    def update_memory(self, action: Dict, success: bool):
        """Update context memory with action result"""
        memory_entry = {
            "action": action,
            "success": success,
            "timestamp": time.time(),
            "screen_state": self.get_screen_state()
        }
        self.context_memory.append(memory_entry)
        if len(self.context_memory) > self.max_memory:
            self.context_memory.pop(0)

    def process_command(self, user_input: str) -> bool:
        """Main command processing pipeline"""
        try:
            # Analyze context
            self.logger.info("Analyzing context...")
            context_analysis = self.analyze_context(user_input)
            if "error" in context_analysis:
                self.logger.error(f"Context analysis failed: {context_analysis['error']}")
                return False
                
            # Create action plan
            self.logger.info("Creating action plan...")
            action_plan = self.create_action_plan(context_analysis)
            if not action_plan:
                self.logger.error("Failed to create action plan")
                return False
                
            # Execute each action in plan
            self.logger.info("Executing actions...")
            for action in action_plan:
                self.logger.info(f"Executing action: {action}")
                success = self.execute_action(action)
                self.update_memory(action, success)
                if not success:
                    self.logger.error(f"Action failed: {action}")
                    return False
                    
            self.logger.info("Command completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Command processing error: {e}")
            return False

    def process_task_queue(self):
        """Background task processor"""
        while True:
            try:
                task = self.task_queue.get()
                self.process_command(task)
                self.task_queue.task_done()
            except Exception as e:
                self.logger.error(f"Queue processing error: {e}")
            time.sleep(0.1)

    def start(self):
        """Start the assistant"""
        task_thread = threading.Thread(target=self.process_task_queue, daemon=True)
        task_thread.start()
        self.logger.info("Dynamic Computer Assistant initialized!")

def run_app():
    """Streamlit interface"""
    st.set_page_config(
        page_title="AI Computer Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 AI Computer Assistant")
    st.write("Your AI-powered computer interaction assistant")

    # Get API key
    API_KEY = "AIzaSyDpaOZq0jE6d4SdTpf1GyNk_lLkB75Kn_8"


    # Initialize assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = DynamicComputerAssistant(API_KEY)
        st.session_state.assistant.start()

    # Command input
    with st.form("command_form"):
        user_input = st.text_input("What would you like me to do?")
        submitted = st.form_submit_button("Execute")

    if submitted and user_input:
        with st.spinner("Processing your request..."):
            success = st.session_state.assistant.process_command(user_input)
            if success:
                st.success("Task completed successfully!")
            else:
                st.error("Failed to complete the task. Please try again.")

    # Display context memory
    if st.session_state.assistant.context_memory:
        st.subheader("Recent Actions")
        for memory in reversed(st.session_state.assistant.context_memory[-5:]):
            with st.expander(f"Action at {time.strftime('%H:%M:%S', time.localtime(memory['timestamp']))}"):
                st.json(memory)

    # System status
    with st.sidebar:
        st.subheader("System Status")
        screen_state = st.session_state.assistant.get_screen_state()
        st.json(screen_state)

if __name__ == "__main__":
    run_app()