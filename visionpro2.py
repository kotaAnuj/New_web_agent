import os
import time
import json
import pyautogui
import google.generativeai as genai
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import cv2
from io import BytesIO
import base64
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
        self.vision_model = genai.GenerativeModel('gemini-1.5-pro-vision')
        
        # System components
        self.task_queue = queue.Queue()
        self.context_memory = []
        self.max_memory = 10
        self.screenshot_buffer = []
        self.running = False
        
        # Initialize PyAutoGUI safely
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.5

    def start(self):
        """Start the assistant's background processes"""
        self.running = True
        self.background_thread = threading.Thread(target=self._background_processor)
        self.background_thread.daemon = True
        self.background_thread.start()

    def stop(self):
        """Stop the assistant's background processes"""
        self.running = False
        if hasattr(self, 'background_thread'):
            self.background_thread.join()

    def _background_processor(self):
        """Background thread for processing tasks"""
        while self.running:
            try:
                if not self.task_queue.empty():
                    task = self.task_queue.get()
                    self.process_command(task)
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Background processor error: {e}")

    def analyze_context(self, user_input: str) -> Dict:
        """Analyze current context and user input"""
        try:
            # Get current screen state
            screen_state = self.capture_and_analyze_screen()
            
            # Prepare context from memory
            context_history = [
                {
                    'action': m['action'],
                    'success': m['success'],
                    'screen_state': m['screen_state'].get('vision_analysis', '')
                }
                for m in self.context_memory[-3:]  # Last 3 actions
            ]
            
            analysis_prompt = f"""
            Analyze this context and user request:
            
            User Input: {user_input}
            Current Screen State: {screen_state.get('vision_analysis', '')}
            Recent Actions: {json.dumps(context_history, indent=2)}
            
            Provide:
            1. Understanding of user's intent
            2. Required preconditions
            3. Potential challenges
            4. Success criteria
            """
            
            analysis = self.analyzer.generate_content(analysis_prompt).text
            
            return {
                'user_input': user_input,
                'screen_state': screen_state,
                'context_history': context_history,
                'analysis': analysis
            }
            
        except Exception as e:
            self.logger.error(f"Context analysis error: {e}")
            return {'error': str(e)}

    def create_action_plan(self, context: Dict) -> List[Dict]:
        """Create detailed action plan based on context"""
        try:
            planning_prompt = f"""
            Create a detailed action plan for this task:
            
            Context: {json.dumps(context, indent=2)}
            
            Generate a list of specific PyAutoGUI actions with:
            1. Exact coordinates or UI element identifiers
            2. Required checks and validations
            3. Error handling steps
            4. Success criteria for each action
            
            Format as a list of JSON objects with 'action', 'params', and 'validation' keys.
            """
            
            plan = self.planner.generate_content(planning_prompt).text
            
            # Parse and validate the plan
            try:
                action_plan = json.loads(plan)
                if not isinstance(action_plan, list):
                    action_plan = [action_plan]
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract actions from text
                actions = plan.split('\n')
                action_plan = []
                for action in actions:
                    if action.strip():
                        action_plan.append({
                            'action': action.strip(),
                            'params': {},
                            'validation': 'Check screen state after action'
                        })
            
            return action_plan
            
        except Exception as e:
            self.logger.error(f"Action planning error: {e}")
            return []

    def capture_and_analyze_screen(self) -> Dict:
        """Capture and analyze the current screen state"""
        try:
            # Capture screen
            screenshot = pyautogui.screenshot()
            
            # Convert to numpy array for OpenCV processing
            screenshot_np = np.array(screenshot)
            screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            
            # Basic image processing
            gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            
            # Find contours (UI elements)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw rectangles around detected UI elements
            screenshot_annotated = screenshot.copy()
            draw = ImageDraw.Draw(screenshot_annotated)
            
            ui_elements = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > 100:  # Filter out tiny elements
                    draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
                    ui_elements.append({
                        'type': 'ui_element',
                        'bounds': {'x': x, 'y': y, 'width': w, 'height': h}
                    })
            
            # Convert to base64 for displaying in Streamlit
            buffered = BytesIO()
            screenshot_annotated.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Save to buffer for context
            self.screenshot_buffer.append({
                'timestamp': time.time(),
                'image': img_str,
                'ui_elements': ui_elements
            })
            
            # Keep only last 5 screenshots
            if len(self.screenshot_buffer) > 5:
                self.screenshot_buffer.pop(0)
            
            # Get additional screen info
            screen_info = pyautogui.size()
            mouse_pos = pyautogui.position()
            
            # Enhanced OCR and text detection
            try:
                # Convert to grayscale for better text detection
                gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)
                # Apply thresholding to get better text contrast
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Use vision model for text detection and analysis
                vision_prompt = """
                Analyze this screenshot and describe:
                1. Main UI elements visible
                2. Current active window/application
                3. Any text that's clearly visible
                4. Interactive elements (buttons, links, etc.)
                5. Overall layout structure
                Be specific about locations and arrangements.
                """
                
                vision_response = self.vision_model.generate_content([
                    vision_prompt,
                    Image.fromarray(cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2RGB))
                ]).text
                
            except Exception as ocr_error:
                self.logger.warning(f"OCR processing error: {ocr_error}")
                vision_response = "Text analysis unavailable"
            
            return {
                'screen_size': {'width': screen_info[0], 'height': screen_info[1]},
                'mouse_position': {'x': mouse_pos[0], 'y': mouse_pos[1]},
                'ui_elements': ui_elements,
                'vision_analysis': vision_response,
                'screenshot': img_str
            }
            
        except Exception as e:
            self.logger.error(f"Screen analysis error: {e}")
            return {}

    def execute_action(self, action: Dict) -> bool:
        """Execute a single atomic action with visual feedback"""
        try:
            # Get pre-action screenshot
            pre_screen = self.capture_and_analyze_screen()
            
            # Generate and execute PyAutoGUI commands
            execution_prompt = f"""
            Generate Python code using PyAutoGUI to execute this action:
            
            Action: {json.dumps(action, indent=2)}
            Screen state: {json.dumps(pre_screen, indent=2)}
            
            Include:
            1. Exact PyAutoGUI commands
            2. Proper error handling
            3. Visual feedback steps
            4. Verification checks
            """
            
            response = self.executor.generate_content(execution_prompt).text
            
            # Extract and execute code
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
            
            # Execute with visual feedback
            st.image(base64.b64decode(pre_screen['screenshot']), caption="Before Action")
            exec(code, safe_globals)
            time.sleep(0.5)
            
            # Get post-action screenshot
            post_screen = self.capture_and_analyze_screen()
            st.image(base64.b64decode(post_screen['screenshot']), caption="After Action")
            
            # Enhanced verification
            verification_prompt = f"""
            Compare these screenshots and verify the action success:
            
            Pre-action state: {json.dumps(pre_screen, indent=2)}
            Post-action state: {json.dumps(post_screen, indent=2)}
            Intended action: {json.dumps(action, indent=2)}
            
            Check:
            1. UI element changes
            2. Mouse position changes
            3. Text/content changes
            4. Overall state changes
            
            Determine if the action achieved its intended effect.
            """
            
            verification = self.analyzer.generate_content(verification_prompt).text
            success = "success" in verification.lower()
            
            # Log the verification result
            if success:
                self.logger.info(f"Action completed successfully: {action}")
            else:
                self.logger.warning(f"Action verification failed: {action}")
                self.logger.debug(f"Verification details: {verification}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            return False

    def update_memory(self, action: Dict, success: bool):
        """Update context memory with visual state"""
        screen_state = self.capture_and_analyze_screen()
        memory_entry = {
            "action": action,
            "success": success,
            "timestamp": time.time(),
            "screen_state": screen_state
        }
        self.context_memory.append(memory_entry)
        if len(self.context_memory) > self.max_memory:
            self.context_memory.pop(0)

    def process_command(self, user_input: str) -> bool:
        """Process command with visual feedback"""
        try:
            # Show initial screen state
            st.write("Current Screen State:")
            initial_screen = self.capture_and_analyze_screen()
            st.image(base64.b64decode(initial_screen['screenshot']), caption="Initial Screen")
            
            # Regular command processing
            context_analysis = self.analyze_context(user_input)
            if "error" in context_analysis:
                st.error(f"Error analyzing context: {context_analysis['error']}")
                return False
                
            action_plan = self.create_action_plan(context_analysis)
            if not action_plan:
                st.error("Failed to create action plan")
                return False
            
            # Display action plan
            st.write("Action Plan:")
            for idx, action in enumerate(action_plan, 1):
                st.write(f"Step {idx}: {action['action']}")
            
            # Execute with visual feedback
            progress_bar = st.progress(0)
            for idx, action in enumerate(action_plan):
                st.write(f"Executing: {action['action']}")
                success = self.execute_action(action)
                self.update_memory(action, success)
                
                if not success:
                    st.error(f"Failed at step {idx + 1}")
                    return False
                    
                progress = (idx + 1) / len(action_plan)
                progress_bar.progress(progress)
            
            # Show final state
            final_screen = self.capture_and_analyze_screen()
            st.write("Final Screen State:")
            st.image(base64.b64decode(final_screen['screenshot']), caption="Final Screen")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Command processing error: {e}")
            st.error(f"Error processing command: {e}")
            return False

def run_app():
    """Enhanced Streamlit interface with visual feedback"""
    st.set_page_config(
        page_title="Visual AI Computer Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Visual AI Computer Assistant")
    st.write("Your AI-powered computer interaction assistant with visual feedback")

    # Get API key
    API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv('GEMINI_API_KEY'))
    if not API_KEY:
        st.error("Please set your Gemini API key in environment variables or Streamlit secrets.")
        return

    # Initialize assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = DynamicComputerAssistant(API_KEY)
        st.session_state.assistant.start()

    # Command input and screen preview
    col1, col2 = st.columns([2, 1])
    
    with col1:
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

    with col2:
        st.subheader("Current Screen")
        current_screen = st.session_state.assistant.capture_and_analyze_screen()
        if current_screen.get('screenshot'):
            st.image(base64.b64decode(current_screen['screenshot']), caption="Live Preview")