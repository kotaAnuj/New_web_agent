import os
import time
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
from typing import Dict, Any

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
        
        # Initialize computer vision
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        self.vision_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        
        # System components
        self.task_queue = queue.Queue()
        self.context_memory = []
        self.max_memory = 10
        
        # Initialize PyAutoGUI safely
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.5

    def analyze_context(self, user_input: str) -> Dict:
        """Analyze user input and current context to understand the request"""
        context_prompt = f"""
        Analyze this user request and current system context.
        Recent actions: {str(self.context_memory[-3:] if self.context_memory else 'None')}
        Current request: {user_input}
        
        Provide analysis in terms of:
        1. User's intent
        2. Required system access
        3. Potential risks
        4. Required resources
        5. Success criteria
        
        Return as structured data.
        """
        
        try:
            analysis = self.analyzer.generate_content(context_prompt).text
            return eval(analysis)
        except Exception as e:
            self.logger.error(f"Context analysis error: {e}")
            return {"error": str(e)}

    def create_action_plan(self, context_analysis: Dict) -> list:
        """Create detailed plan of actions based on context analysis"""
        planning_prompt = f"""
        Create a detailed action plan based on this context:
        {str(context_analysis)}
        
        Break down into atomic actions that can be executed by PyAutoGUI.
        Consider system state and required timing between actions.
        Include error checking steps.
        """
        
        try:
            plan = self.planner.generate_content(planning_prompt).text
            return eval(plan)
        except Exception as e:
            self.logger.error(f"Planning error: {e}")
            return []

    def execute_action(self, action: Dict):
        """Execute a single atomic action"""
        execution_prompt = f"""
        Execute this atomic action safely:
        {str(action)}
        
        Current screen state: {self.get_screen_state()}
        Determine exact PyAutoGUI commands needed.
        """
        
        try:
            commands = self.executor.generate_content(execution_prompt).text
            exec(commands)  # Execute the generated PyAutoGUI commands
            time.sleep(0.5)  # Safety delay
            return True
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            return False

    def get_screen_state(self) -> Dict:
        """Analyze current screen state"""
        try:
            # Capture screen
            screenshot = pyautogui.screenshot()
            screen_np = np.array(screenshot)
            screen_pil = Image.fromarray(screen_np)
            
            # Process with vision model
            inputs = self.feature_extractor(images=screen_pil, return_tensors="pt")
            with torch.no_grad():
                outputs = self.vision_model(**inputs)
            
            # Get screen elements and their locations
            screen_info = pyautogui.size()
            active_window = pyautogui.getActiveWindow()
            mouse_position = pyautogui.position()
            
            return {
                "resolution": screen_info,
                "active_window": str(active_window) if active_window else None,
                "mouse_position": mouse_position,
                "vision_analysis": outputs.logits.softmax(-1).tolist()
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
            context_analysis = self.analyze_context(user_input)
            if "error" in context_analysis:
                return False
                
            # Create action plan
            action_plan = self.create_action_plan(context_analysis)
            if not action_plan:
                return False
                
            # Execute each action in plan
            for action in action_plan:
                success = self.execute_action(action)
                self.update_memory(action, success)
                if not success:
                    return False
                    
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
    assistant = DynamicComputerAssistant(API_KEY)
    assistant.start()

    # Command input
    with st.form("command_form"):
        user_input = st.text_input("What would you like me to do?")
        submitted = st.form_submit_button("Execute")

    if submitted and user_input:
        with st.spinner("Processing your request..."):
            success = assistant.process_command(user_input)
            if success:
                st.success("Task completed successfully!")
            else:
                st.error("Failed to complete the task. Please try again.")

    # Display context memory
    if assistant.context_memory:
        st.subheader("Recent Actions")
        for memory in reversed(assistant.context_memory[-5:]):
            with st.expander(f"Action at {time.strftime('%H:%M:%S', time.localtime(memory['timestamp']))}"):
                st.json(memory)

    # System status
    with st.sidebar:
        st.subheader("System Status")
        screen_state = assistant.get_screen_state()
        st.json(screen_state)

if __name__ == "__main__":
    run_app()