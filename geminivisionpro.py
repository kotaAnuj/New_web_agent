import os
import time
import json
import pyautogui
import google.generativeai as genai
import torch
from torch import nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
    ViTFeatureExtractor,
    AutoModelForObjectDetection,
    DetrFeatureExtractor,
    AutoProcessor,
    CLIPProcessor, 
    CLIPModel
)
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
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class TransformerModels:
    """Container for transformer models"""
    text_model: AutoModelForSequenceClassification
    vision_model: AutoModelForImageClassification
    object_detector: AutoModelForObjectDetection
    clip_model: CLIPModel
    text_tokenizer: AutoTokenizer
    vision_processor: ViTFeatureExtractor
    object_processor: DetrFeatureExtractor
    clip_processor: CLIPProcessor

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
        
        # Initialize transformer models
        self.transformer_models = self._initialize_transformers()
        
        # System components
        self.task_queue = queue.Queue()
        self.context_memory = []
        self.max_memory = 10
        self.screenshot_buffer = []
        self.is_running = False
        self.executor_pool = ThreadPoolExecutor(max_workers=3)
        
        # Initialize PyAutoGUI safely
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.5

        # Cache for transformer predictions
        self.prediction_cache = {}
        self.cache_ttl = 300  # 5 minutes

    def _initialize_transformers(self) -> TransformerModels:
        """Initialize all transformer models"""
        try:
            # Text classification model for command understanding
            text_model_name = "bert-base-uncased"
            text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            text_model = AutoModelForSequenceClassification.from_pretrained(
                text_model_name,
                num_labels=3  # command, query, description
            )

            # Vision model for screen analysis
            vision_model_name = "google/vit-base-patch16-224"
            vision_processor = ViTFeatureExtractor.from_pretrained(vision_model_name)
            vision_model = AutoModelForImageClassification.from_pretrained(vision_model_name)

            # Object detection model for UI elements
            object_model_name = "facebook/detr-resnet-50"
            object_processor = DetrFeatureExtractor.from_pretrained(object_model_name)
            object_detector = AutoModelForObjectDetection.from_pretrained(object_model_name)

            # CLIP model for combined vision-language tasks
            clip_model_name = "openai/clip-vit-base-patch32"
            clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            clip_model = CLIPModel.from_pretrained(clip_model_name)

            # Move models to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            text_model.to(device)
            vision_model.to(device)
            object_detector.to(device)
            clip_model.to(device)

            return TransformerModels(
                text_model=text_model,
                vision_model=vision_model,
                object_detector=object_detector,
                clip_model=clip_model,
                text_tokenizer=text_tokenizer,
                vision_processor=vision_processor,
                object_processor=object_processor,
                clip_processor=clip_processor
            )

        except Exception as e:
            self.logger.error(f"Error initializing transformers: {e}")
            raise

    def start(self):
        """Start the assistant"""
        if not self.is_running:
            self.is_running = True
            self.processor_thread = threading.Thread(target=self._process_tasks)
            self.processor_thread.daemon = True
            self.processor_thread.start()
            self.logger.info("Assistant started successfully")

    def stop(self):
        """Stop the assistant"""
        self.is_running = False
        self.executor_pool.shutdown(wait=True)
        self.logger.info("Assistant stopped")

    def _process_tasks(self):
        """Process tasks from the queue"""
        while self.is_running:
            try:
                if not self.task_queue.empty():
                    task = self.task_queue.get()
                    self._execute_task(task)
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Task processing error: {e}")

    def _execute_task(self, task: Dict):
        """Execute a single task"""
        try:
            # Get task execution plan
            plan = self._get_execution_plan(task)
            
            # Execute each step in the plan
            for step in plan['steps']:
                action = step['action']
                params = step.get('params', {})
                
                if action == 'move':
                    pyautogui.moveTo(params['x'], params['y'])
                elif action == 'click':
                    pyautogui.click(params.get('x'), params.get('y'))
                elif action == 'type':
                    pyautogui.typewrite(params['text'])
                elif action == 'keypress':
                    pyautogui.press(params['key'])
                
                time.sleep(0.5)  # Brief pause between actions
                
            # Update context memory
            self._update_memory(task, plan)
            
        except Exception as e:
            self.logger.error(f"Task execution error: {e}")

    def _get_execution_plan(self, task: Dict) -> Dict:
        """Generate execution plan for a task"""
        try:
            # Analyze current screen state
            screen_state = self.capture_and_analyze_screen()
            
            # Generate plan using Gemini
            plan_prompt = f"""
            Task: {task['command']}
            Current screen state: {screen_state}
            Recent actions: {self.context_memory[-3:] if self.context_memory else 'None'}
            
            Create a detailed plan with these steps:
            1. Required mouse movements
            2. Click actions needed
            3. Text input if required
            4. Any keyboard shortcuts
            5. Verification steps
            
            Format as JSON with 'steps' array containing actions and parameters.
            """
            
            plan_response = self.planner.generate_content(plan_prompt).text
            return json.loads(plan_response)
            
        except Exception as e:
            self.logger.error(f"Plan generation error: {e}")
            return {"steps": []}

    def _update_memory(self, task: Dict, plan: Dict):
        """Update context memory"""
        memory_entry = {
            'timestamp': time.time(),
            'task': task,
            'plan': plan,
            'screen_state': self.screenshot_buffer[-1] if self.screenshot_buffer else None
        }
        
        self.context_memory.append(memory_entry)
        if len(self.context_memory) > self.max_memory:
            self.context_memory.pop(0)

    def _process_text_with_transformer(self, text: str) -> Dict[str, float]:
        """Process text input using BERT"""
        try:
            # Check cache first
            cache_key = f"text_{text}"
            if cache_key in self.prediction_cache:
                cache_time, prediction = self.prediction_cache[cache_key]
                if time.time() - cache_time < self.cache_ttl:
                    return prediction

            inputs = self.transformer_models.text_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.transformer_models.text_model.device)

            with torch.no_grad():
                outputs = self.transformer_models.text_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            prediction = {
                "command_probability": probs[0][0].item(),
                "query_probability": probs[0][1].item(),
                "description_probability": probs[0][2].item()
            }

            # Cache the prediction
            self.prediction_cache[cache_key] = (time.time(), prediction)
            return prediction

        except Exception as e:
            self.logger.error(f"Text processing error: {e}")
            return {}

    def _process_image_with_transformers(self, image: Image.Image) -> Dict[str, Any]:
        """Process image using multiple transformer models"""
        try:
            # Check cache first
            image_hash = hash(image.tobytes())
            cache_key = f"image_{image_hash}"
            if cache_key in self.prediction_cache:
                cache_time, prediction = self.prediction_cache[cache_key]
                if time.time() - cache_time < self.cache_ttl:
                    return prediction

            # Vision transformer processing
            vision_inputs = self.transformer_models.vision_processor(
                images=image, 
                return_tensors="pt"
            ).to(self.transformer_models.vision_model.device)

            # Object detection processing
            object_inputs = self.transformer_models.object_processor(
                images=image, 
                return_tensors="pt"
            ).to(self.transformer_models.object_detector.device)

            # CLIP processing
            clip_inputs = self.transformer_models.clip_processor(
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.transformer_models.clip_model.device)

            with torch.no_grad():
                # Vision classification
                vision_outputs = self.transformer_models.vision_model(**vision_inputs)
                vision_probs = torch.nn.functional.softmax(vision_outputs.logits, dim=-1)

                # Object detection
                object_outputs = self.transformer_models.object_detector(**object_inputs)
                
                # CLIP image features
                clip_outputs = self.transformer_models.clip_model.get_image_features(**clip_inputs)

            # Process results
            prediction = {
                "scene_classification": vision_probs[0].tolist(),
                "detected_objects": self._process_object_detection(object_outputs),
                "image_features": clip_outputs[0].tolist()
            }

            # Cache the prediction
            self.prediction_cache[cache_key] = (time.time(), prediction)
            return prediction

        except Exception as e:
            self.logger.error(f"Image processing error: {e}")
            return {}

    def _process_object_detection(self, outputs) -> List[Dict[str, Any]]:
        """Process object detection outputs"""
        try:
            probas = outputs.logits.softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.9
            
            boxes = outputs.pred_boxes[0, keep].cpu()
            probas = probas[keep].cpu()
            
            return [
                {
                    "box": box.tolist(),
                    "probability": prob.max().item(),
                    "label": prob.argmax().item()
                }
                for box, prob in zip(boxes, probas)
            ]
        except Exception as e:
            self.logger.error(f"Object detection processing error: {e}")
            return []

    def analyze_context(self, user_input: str) -> Dict:
        """Analyze current context with transformer support"""
        try:
            screen_state = self.capture_and_analyze_screen()
            
            # Process text with transformer
            text_analysis = self._process_text_with_transformer(user_input)
            
            # Combine with Gemini analysis
            context_prompt = f"""
            Analyze this situation:
            User input: {user_input}
            Text analysis: {text_analysis}
            Current screen state: {screen_state}
            Recent actions: {self.context_memory[-3:] if self.context_memory else 'None'}
            
            Provide:
            1. Task interpretation
            2. Required actions
            3. Potential challenges
            4. Success criteria
            """
            
            analysis = self.analyzer.generate_content(context_prompt).text
            
            return {
                'analysis': analysis,
                'text_analysis': text_analysis,
                'screen_state': screen_state,
                'user_input': user_input
            }
            
        except Exception as e:
            self.logger.error(f"Context analysis error: {e}")
            return {"error": str(e)}

    def capture_and_analyze_screen(self) -> Dict:
        """Capture and analyze screen with transformer support"""
        try:
            # Capture screen
            screenshot = pyautogui.screenshot()
            
            # Process with transformers
            transformer_analysis = self._process_image_with_transformers(screenshot)
            
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
            
            # Combine transformer object detection with contour detection
            ui_elements = []
            detected_objects = transformer_analysis.get('detected_objects', [])
            
            # Process detected objects
            for obj in detected_objects:
                x1, y1, x2, y2 = [int(coord) for coord in obj['box']]
                draw.rectangle([x1, y1, x2, y2], outline='blue', width=2)
                ui_elements.append({
                    'type': 'detected_object',
                    'bounds': {'x': x1, 'y': y1, 'width': x2-x1, 'height': y2-y1},
                    'probability': obj['probability'],
                    'label': obj['label']
                })
    
            # Process contours
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
                'ui_elements': ui_elements,
                'transformer_analysis': transformer_analysis
            })
            
            # Keep only last 5 screenshots
            if len(self.screenshot_buffer) > 5:
                self.screenshot_buffer.pop(0)
            
            # Get additional screen info
            screen_info = pyautogui.size()
            mouse_pos = pyautogui.position()
            
            # Use vision model to analyze screenshot
            vision_prompt = """
            Analyze this screenshot and describe:
            1. Main UI elements visible
            2. Current active window/application
            3. Any text that's clearly visible
            4. Interactive elements (buttons, links, etc.)
            5. Overall layout structure
            """
            
            vision_response = self.vision_model.generate_content([
                vision_prompt,
                Image.fromarray(screenshot_np)
            ]).text
            
            return {
                'screen_size': {'width': screen_info[0], 'height': screen_info[1]},
                'mouse_position': {'x': mouse_pos[0], 'y': mouse_pos[1]},
                'ui_elements': ui_elements,
                'vision_analysis': vision_response,
                'transformer_analysis': transformer_analysis,
                'screenshot': img_str
            }
            
        except Exception as e:
            self.logger.error(f"Screen analysis error: {e}")
            return {}

    def execute_command(self, command: str) -> Dict:
        """Execute a user command"""
        try:
            # Analyze command context
            context = self.analyze_context(command)
            
            # Create task
            task = {
                'command': command,
                'context': context,
                'timestamp': time.time()
            }
            
            # Add to queue
            self.task_queue.put(task)
            
            return {
                'status': 'queued',
                'task_id': id(task),
                'analysis': context
            }
            
        except Exception as e:
            self.logger.error(f"Command execution error: {e}")
            return {'status': 'error', 'message': str(e)}

def run_app():
    """Enhanced Streamlit interface with transformer support"""
    st.set_page_config(
        page_title="Visual AI Computer Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Visual AI Computer Assistant")
    st.write("Your AI-powered computer interaction assistant with transformer support")

    # Session state initialization
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.show_memory = False
        st.session_state.show_debug = False
        st.session_state.show_transformer_debug = False

    # Get API key
    API_KEY = "AIzaSyDpaOZq0jE6d4SdTpf1GyNk_lLkB75Kn_8"

    # Initialize assistant
    if not st.session_state.initialized:
        with st.spinner("Initializing transformer models..."):
            st.session_state.assistant = DynamicComputerAssistant(API_KEY)
            st.session_state.assistant.start()
            st.session_state.initialized = True

    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Command input
        with st.form("command_form"):
            command = st.text_input("Enter command:", placeholder="Type your command here...")
            submitted = st.form_submit_button("Execute")
            
            if submitted and command:
                with st.spinner("Processing command..."):
                    result = st.session_state.assistant.execute_command(command)
                    
                if result.get('status') == 'error':
                    st.error(result['message'])
                else:
                    st.success("Command queued for execution")
                    if 'analysis' in result:
                        with st.expander("Command Analysis", expanded=True):
                            st.json(result['analysis'])

        # Latest screenshot
        if st.session_state.assistant.screenshot_buffer:
            latest = st.session_state.assistant.screenshot_buffer[-1]
            st.image(latest['image'], caption="Current Screen State", use_column_width=True)

    with col2:
        # Debug controls
        with st.expander("Debug Controls"):
            st.session_state.show_memory = st.checkbox("Show Memory")
            st.session_state.show_debug = st.checkbox("Show Debug Info")
            st.session_state.show_transformer_debug = st.checkbox("Show Transformer Analysis")
            
            if st.button("Stop Assistant"):
                st.session_state.assistant.stop()
                st.session_state.initialized = False
                st.experimental_rerun()

        # Memory viewer
        if st.session_state.show_memory:
            with st.expander("Context Memory", expanded=True):
                for memory in reversed(st.session_state.assistant.context_memory):
                    st.write(f"Task: {memory['task']['command']}")
                    st.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(memory['timestamp']))}")
                    if st.button("Show Details", key=f"memory_{id(memory)}"):
                        st.json(memory)

        # Debug info
        if st.session_state.show_debug:
            with st.expander("Debug Information", expanded=True):
                st.write("Queue Size:", st.session_state.assistant.task_queue.qsize())
                st.write("Memory Size:", len(st.session_state.assistant.context_memory))
                st.write("Screenshot Buffer Size:", len(st.session_state.assistant.screenshot_buffer))

        # Transformer analysis
        if st.session_state.show_transformer_debug and st.session_state.assistant.screenshot_buffer:
            with st.expander("Transformer Analysis", expanded=True):
                latest = st.session_state.assistant.screenshot_buffer[-1]
                st.json(latest['transformer_analysis'])

if __name__ == "__main__":
    run_app()                                   