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
from typing import Dict, Any

class ComputerAssistant:
    def __init__(self, api_key: str):
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        # Configure Gemini API
        genai.configure(api_key=api_key)
        self.language_model = genai.GenerativeModel('gemini-1.5-pro')  # Changed to pro for better parsing

        # Computer vision components
        try:
            self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
            self.vision_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        except Exception as e:
            self.logger.error(f"Error loading vision models: {e}")
            self.feature_extractor = None
            self.vision_model = None

        # Task execution components
        self.task_queue = queue.Queue()
        self.context_memory = []
        self.max_context_length = 10

    def natural_language_parser(self, command: str) -> Dict[str, Any]:
        system_prompt = """
        Parse the following command into a JSON structure. Response must be valid JSON.
        Available actions are: open_app, type_text, click, search, screenshot.
        Format:
        {
            "action_type": "action_name",
            "target": "target_name",
            "parameters": {"param1": "value1"}
        }
        """
        try:
            response = self.language_model.generate_content([
                {"role": "user", "parts": [f"{system_prompt}\n\nCommand: {command}"]}
            ])
            # Extract JSON string from response and parse it
            json_str = response.text.strip().replace("```json", "").replace("```", "")
            parsed_command = json.loads(json_str)
            return parsed_command
        except Exception as e:
            self.logger.error(f"Command parsing error: {e}")
            return {"action_type": "error", "message": str(e)}

    def analyze_screen(self) -> Dict[str, Any]:
        if not self.feature_extractor or not self.vision_model:
            return {}
            
        try:
            # Take screenshot and convert to RGB
            screen = pyautogui.screenshot()
            screen_np = np.array(screen)
            screen_pil = Image.fromarray(screen_np)
            
            # Prepare image for model
            inputs = self.feature_extractor(images=screen_pil, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.vision_model(**inputs)
            
            # Process outputs
            predictions = outputs.logits.softmax(-1)
            return {"screen_elements": predictions.tolist()}
            
        except Exception as e:
            self.logger.error(f"Screen analysis error: {e}")
            return {}

    def execute_task(self, task: Dict[str, Any]):
        action_type = task.get('action_type', '')
        target = task.get('target', '')
        parameters = task.get('parameters', {})

        try:
            if action_type == 'open_app':
                pyautogui.press('win')
                time.sleep(0.5)
                pyautogui.typewrite(target)
                time.sleep(0.5)
                pyautogui.press('enter')

            elif action_type == 'type_text':
                text = parameters.get('text', '')
                pyautogui.typewrite(text)

            elif action_type == 'click':
                x, y = parameters.get('coordinates', (0, 0))
                pyautogui.moveTo(x, y, duration=0.5)
                pyautogui.click()

            elif action_type == 'search':
                # Open browser and perform search
                pyautogui.press('win')
                time.sleep(0.5)
                pyautogui.typewrite('chrome')
                pyautogui.press('enter')
                time.sleep(2)
                pyautogui.hotkey('ctrl', 'l')
                search_query = target.replace(' ', '+')
                pyautogui.typewrite(f"https://www.google.com/search?q={search_query}")
                pyautogui.press('enter')

            elif action_type == 'screenshot':
                screenshot = pyautogui.screenshot()
                save_path = parameters.get('path', 'screenshot.png')
                screenshot.save(save_path)
                st.image(save_path, caption="Screenshot")

            self.update_context_memory(task)
            time.sleep(0.5)  # Add small delay between actions

        except Exception as e:
            self.logger.error(f"Task execution error: {e}")
            raise

    def update_context_memory(self, task: Dict[str, Any]):
        self.context_memory.append(task)
        if len(self.context_memory) > self.max_context_length:
            self.context_memory.pop(0)

    def process_task_queue(self):
        while True:
            try:
                task = self.task_queue.get()
                if task['action_type'] != 'error':
                    self.execute_task(task)
                self.task_queue.task_done()
            except Exception as e:
                self.logger.error(f"Queue processing error: {e}")
            time.sleep(0.1)  # Prevent CPU hogging

    def start(self):
        task_thread = threading.Thread(target=self.process_task_queue, daemon=True)
        task_thread.start()
        self.logger.info("Computer Assistant initialized and ready!")

    def process_command(self, command: str) -> bool:
        parsed_task = self.natural_language_parser(command)
        if parsed_task['action_type'] != 'error':
            self.task_queue.put(parsed_task)
            return True
        else:
            self.logger.warning(f"Could not parse command: {command}")
            return False

def run_app():
    st.set_page_config(page_title="Computer Interaction Assistant")

    st.title("Computer Interaction Assistant")
    st.write("Enter a command to interact with your computer")

    # Get API key from environment or Streamlit secrets
    API_KEY = "AIzaSyDpaOZq0jE6d4SdTpf1GyNk_lLkB75Kn_8"
    
    

    assistant = ComputerAssistant(API_KEY)
    assistant.start()

    user_input = st.text_input("Enter command:", key="command_input")

    if st.button("Execute") and user_input:
        try:
            with st.spinner("Executing command..."):
                success = assistant.process_command(user_input)
                if success:
                    st.success("Command executed successfully!")
                else:
                    st.error("Failed to parse command. Please try rephrasing.")
                time.sleep(1)  # Wait for tasks to complete
                assistant.analyze_screen()
        except Exception as e:
            st.error(f"Error executing command: {e}")

    if assistant.context_memory:
        st.write("Recent commands:")
        for task in reversed(assistant.context_memory):
            st.write(f"- {task['action_type']}: {task.get('target', '')}")

if __name__ == "__main__":
    run_app()