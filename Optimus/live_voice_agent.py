import asyncio
import base64
import io
import os
import sys
import traceback
import argparse
import cv2
import pyaudio
import PIL.Image
import mss
import webbrowser
import subprocess
import json
from google import genai

# For backward compatibility with older Python versions
if sys.version_info < (3, 11, 0):
    import taskgroup
    import exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# Model Choice: Use latest native audio model for Live API
MODEL = "gemini-2.5-flash-native-audio-latest"
DEFAULT_MODE = "none"

# System Instruction for the Agent
SYSTEM_INSTRUCTION = """
You are a helpful and friendly live voice assistant named J.A.R.V.I.S.
Your most important rule: Always respond to the user in the SAME language they use.
If they speak or write in Persian, respond in Persian. If they use English, respond in English.
If they speak Arabic, respond in Arabic.
Keep your responses concise and natural for a voice conversation.
Do not share your internal reasoning, "thought" processes, or planning steps with the user.
You have access to tools for file management, app launching, and web browsing. Use them when requested.
"""

# Initialize Client
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    sys.exit(1)

client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

# --- Tool Definitions ---
create_folder = {
    "name": "create_folder",
    "description": "Creates a new folder at the specified path relative to the script's root directory.",
    "parameters": {
        "type": "OBJECT",
        "properties": { "folder_path": { "type": "STRING", "description": "The path for the new folder."}},
        "required": ["folder_path"]
    }
}

create_file = {
    "name": "create_file",
    "description": "Creates a new file with specified content at a given path.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "file_path": { "type": "STRING", "description": "The path for the new file."},
            "content": { "type": "STRING", "description": "The content to write into the new file."}
        },
        "required": ["file_path", "content"]
    }
}

edit_file = {
    "name": "edit_file",
    "description": "Appends content to an existing file at a specified path.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "file_path": { "type": "STRING", "description": "The path of the file to edit."},
            "content": { "type": "STRING", "description": "The content to append to the file."}
        },
        "required": ["file_path", "content"]
    }
}

list_files = {
    "name": "list_files",
    "description": "Lists all files and directories within a specified folder.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "directory_path": { "type": "STRING", "description": "The path to inspect. Defaults to '.'."}
        }
    }
}

read_file = {
    "name": "read_file",
    "description": "Reads the entire content of a specified file.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "file_path": { "type": "STRING", "description": "The path of the file to read."}
        },
        "required": ["file_path"]
    }
}

open_application = {
    "name": "open_application",
    "description": "Opens or launches a desktop application.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "application_name": { "type": "STRING", "description": "The name of the app (e.g., 'Notepad', 'Calculator')."}
        },
        "required": ["application_name"]
    }
}

open_website = {
    "name": "open_website",
    "description": "Opens a given URL in the default web browser.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "url": { "type": "STRING", "description": "The URL to open."}
        },
        "required": ["url"]
    }
}

tools = [
    {'google_search': {}}, 
    {'code_execution': {}}, 
    {"function_declarations": [create_folder, create_file, edit_file, list_files, read_file, open_application, open_website]}
]

CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
    "tools": tools
}

pya = pyaudio.PyAudio()

class LiveVoiceAgent:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        self.audio_in_queue = None
        self.out_queue = None
        self.ui_queue = None
        self.session = None
        self.audio_stream = None
        
        # Determine best devices (Bluetooth Hands-Free vs Default)
        self.in_idx, self.out_idx = self._find_best_devices()

    def _find_best_devices(self):
        """Find the best audio devices (prefer Hands-Free for Bluetooth headsets)."""
        in_idx = None
        out_idx = None
        
        print("\nScanning for audio devices...")
        for i in range(pya.get_device_count()):
            info = pya.get_device_info_by_index(i)
            name = info['name'].lower()
            
            # Look for Hands-Free or AG Audio which supports duplex on BT
            if 'hands-free' in name or 'ag audio' in name:
                if info['maxInputChannels'] > 0 and in_idx is None:
                    in_idx = i
                    print(f"  Found Hands-Free Input: {info['name']} (Index {i})")
                if info['maxOutputChannels'] > 0 and out_idx is None:
                    out_idx = i
                    print(f"  Found Hands-Free Output: {info['name']} (Index {i})")
        
        # Fallback to defaults if not found
        if in_idx is None:
            default_in = pya.get_default_input_device_info()
            in_idx = default_in['index']
            print(f"  Using Default Input: {default_in['name']} (Index {in_idx})")
        if out_idx is None:
            default_out = pya.get_default_output_device_info()
            out_idx = default_out['index']
            print(f"  Using Default Output: {default_out['name']} (Index {out_idx})")
            
        return in_idx, out_idx

    # --- Tool Implementation Methods ---
    def _create_folder(self, folder_path):
        try:
            if not folder_path: return {"status": "error", "message": "Invalid path."}
            if os.path.exists(folder_path): return {"status": "skipped", "message": "Folder exists."}
            os.makedirs(folder_path)
            return {"status": "success", "message": f"Created folder '{folder_path}'."}
        except Exception as e: return {"status": "error", "message": str(e)}

    def _create_file(self, file_path, content):
        try:
            if not file_path: return {"status": "error", "message": "Invalid path."}
            with open(file_path, 'w') as f: f.write(content)
            return {"status": "success", "message": f"Created file '{file_path}'."}
        except Exception as e: return {"status": "error", "message": str(e)}

    def _edit_file(self, file_path, content):
        try:
            if not os.path.exists(file_path): return {"status": "error", "message": "File not found."}
            with open(file_path, 'a') as f: f.write(f"\n{content}")
            return {"status": "success", "message": f"Appended to '{file_path}'."}
        except Exception as e: return {"status": "error", "message": str(e)}

    def _list_files(self, directory_path='.'):
        try:
            path = directory_path if directory_path else '.'
            if not os.path.isdir(path): return {"status": "error", "message": "Invalid directory."}
            files = os.listdir(path)
            return {"status": "success", "files": files}
        except Exception as e: return {"status": "error", "message": str(e)}

    def _read_file(self, file_path):
        try:
            if not os.path.isfile(file_path): return {"status": "error", "message": "File not found."}
            with open(file_path, 'r') as f: content = f.read()
            return {"status": "success", "content": content}
        except Exception as e: return {"status": "error", "message": str(e)}

    def _open_application(self, application_name):
        try:
            if not application_name: return {"status": "error", "message": "Invalid name."}
            if sys.platform == "win32":
                app_map = {"calculator": "calc:", "notepad": "notepad", "chrome": "chrome", "firefox": "firefox", "explorer": "explorer"}
                cmd = app_map.get(application_name.lower(), application_name)
                subprocess.Popen(f"start {cmd}", shell=True)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-a", application_name])
            else:
                subprocess.Popen([application_name.lower()])
            return {"status": "success", "message": f"Opened '{application_name}'."}
        except Exception as e: return {"status": "error", "message": str(e)}

    def _open_website(self, url):
        try:
            if not url.startswith(('http://', 'https://')): url = 'https://' + url
            webbrowser.open(url)
            return {"status": "success", "message": f"Opened '{url}'."}
        except Exception as e: return {"status": "error", "message": str(e)}

    async def send_text(self):
        """Task to send text messages from console input."""
        while True:
            text = await asyncio.to_thread(input, "Message (or 'q' to quit) > ")
            if text.lower() == "q":
                break
            if text:
                # Use correct 'turns' parameter for send_client_content
                await self.session.send_client_content(
                    turns=[{"role": "user", "parts": [{"text": text}]}],
                    turn_complete=True
                )

    def _get_frame(self, cap):
        """Helper to capture and process camera frames."""
        ret, frame = cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        return {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_io.read()).decode()
        }

    async def get_frames(self):
        """Task to stream camera frames."""
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)
        cap.release()

    def _get_screen(self):
        """Helper to capture screen content."""
        sct = mss.mss()
        monitor = sct.monitors[1] # Use primary monitor
        i = sct.grab(monitor)
        
        img = PIL.Image.frombytes("RGB", i.size, i.bgra, "raw", "BGRX")
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        return {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_io.read()).decode()
        }

    async def get_screen(self):
        """Task to stream screen content."""
        while True:
            frame = await asyncio.to_thread(self._get_screen)
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

    async def send_realtime(self):
        """Task to send multimodal content from the output queue."""
        while True:
            msg = await self.out_queue.get()
            # Use specific parameters for streaming audio/video chunks
            mime = msg.get("mime_type", "")
            if "audio" in mime:
                await self.session.send_realtime_input(audio=msg)
            else:
                await self.session.send_realtime_input(video=msg)

    async def listen_audio(self):
        """Task to capture audio from microphone."""
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=self.in_idx,
            frames_per_buffer=CHUNK_SIZE,
        )
        
        while True:
            # exception_on_overflow=False helps prevent crashes on slow processing
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, exception_on_overflow=False)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        """Task to receive audio and text responses from Gemini."""
        while True:
            try:
                # The SDK's session.receive() returns an async iterator of responses
                async for response in self.session.receive():
                    # Handle Tool Calls
                    if response.tool_call:
                        function_responses = []
                        for fc in response.tool_call.function_calls:
                            name = fc.name
                            args = fc.args
                            result = {}
                            
                            # Notify UI
                            if self.ui_queue:
                                self.ui_queue.put(("text", f"[Tool: {name}]"))

                            print(f"\n[Tool Call] {name}({args})")

                            if name == "create_folder": result = self._create_folder(args.get("folder_path"))
                            elif name == "create_file": result = self._create_file(args.get("file_path"), args.get("content"))
                            elif name == "edit_file": result = self._edit_file(args.get("file_path"), args.get("content"))
                            elif name == "list_files": result = self._list_files(args.get("directory_path"))
                            elif name == "read_file": result = self._read_file(args.get("file_path"))
                            elif name == "open_application": result = self._open_application(args.get("application_name"))
                            elif name == "open_website": result = self._open_website(args.get("url"))
                            
                            function_responses.append({
                                "id": fc.id,
                                "name": name,
                                "response": result
                            })
                        
                        await self.session.send_tool_response(function_responses=function_responses)
                        continue

                    # If response.data is available, it's the simplest way to get audio bytes
                    data = getattr(response, 'data', None)
                    if data:
                        self.audio_in_queue.put_nowait(data)
                        continue
                        
                    # Detailed check for server_content for multimodal/multipart data
                    if hasattr(response, 'server_content') and response.server_content:
                        model_turn = response.server_content.model_turn
                        if model_turn and model_turn.parts:
                            for part in model_turn.parts:
                                inline_data = getattr(part, 'inline_data', None)
                                if inline_data:
                                    audio_bytes = inline_data.data
                                    if audio_bytes:
                                        self.audio_in_queue.put_nowait(audio_bytes)
                                elif hasattr(part, 'text') and part.text:
                                    # Filter out internal reasoning (thoughts) often wrapped in asterisks
                                    clean_text = part.text.strip()
                                    if not clean_text.startswith("**") and clean_text:
                                        print(part.text, end="", flush=True)
                                        if self.ui_queue:
                                            self.ui_queue.put(("text", clean_text))

                # Clear queue on turn transition/interruption
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()
            except Exception as e:
                print(f"\n[Receive Error]: {e}")
                # traceback.print_exc()
                break

    async def play_audio(self):
        """Task to play received audio chunks."""
        try:
            stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
                output_device_index=self.out_idx,
            )
            
            while True:
                bytestream = await self.audio_in_queue.get()
                if bytestream:
                    await asyncio.to_thread(stream.write, bytestream)
        except Exception as e:
            print(f"\n[Playback Error]: {e}")

    async def run(self):
        """Main loop to orchestrate all tasks."""
        print(f"Connecting to {MODEL}...")
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                if self.ui_queue:
                    self.ui_queue.put(("status", "Connected to Gemini"))

                print("\nConnected! You can start talking now.")
                print("Important: Use headphones to prevent echo.\n")

                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                tg.create_task(self.send_realtime())
                
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                    print("Streaming camera content...")
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())
                    print("Streaming screen content...")

                await self.send_text()
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            print("\nExiting...")
        except Exception as e:
            print(f"\nError occuried: {e}")
            traceback.print_exc()
        finally:
            if self.audio_stream:
                self.audio_stream.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Live Voice Agent")
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["camera", "screen", "none"],
        help="Multimodal input mode: camera, screen, or none (default)"
    )
    args = parser.parse_args()

    agent = LiveVoiceAgent(video_mode=args.mode)
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        pass
