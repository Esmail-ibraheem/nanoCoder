import pygame
import pyaudio
import struct
from PIL import Image, ImageSequence
import datetime
import threading
import os
import cv2
import mediapipe as mp
import numpy as np
import sys
import queue
import asyncio
import calendar

# Initialize pygame
pygame.init()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add parent directory to path to import live_voice_agent
sys.path.append(os.path.join(BASE_DIR, '..'))
try:
    from live_voice_agent import LiveVoiceAgent
except ImportError:
    print("Could not import live_voice_agent. Make sure it is in the parent directory.")
    sys.exit(1)

# Load custom font (Orbitron)
font_path = os.path.join(BASE_DIR, 'Orbitron-VariableFont_wght.ttf')
clock_font = pygame.font.Font(font_path, 80)
calendar_font = pygame.font.Font(font_path, 20)
description_font = pygame.font.Font(font_path, 16)
 

track_font = pygame.font.SysFont("SF Mono", 18)

ENABLE_HAND_TRACKING = True  # Set to True or False for hand control


hand_landmarks_global = None
hand_closed_global = False
wrist_screen_pos = (0, 0)
grab_active = False

WHITE = (255, 255, 255)
CYAN = WHITE # For backward compatibility in this file if missed any
BLACK = (0, 0, 0)
HIGHLIGHT_ALPHA = 80

todo_file_path = os.path.join(BASE_DIR, 'todo.txt') 
todo_font = pygame.font.Font(font_path, 30)

def load_todo_tasks():
    if os.path.exists(todo_file_path):
        with open(todo_file_path, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return []

# Screen setup
info = pygame.display.Info()
screen_width, screen_height = info.current_w, info.current_h
screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
pygame.display.set_caption('O.P.T.I.M.U.S')

# Load JARVIS face GIF (Legacy/Default)
gif_path = os.path.join(BASE_DIR, 'jarvis.gif')
gif = Image.open(gif_path)
frames = [frame.copy().convert("RGBA") for frame in ImageSequence.Iterator(gif)]
frame_surfaces = [pygame.image.frombuffer(frame.tobytes(), frame.size, "RGBA") for frame in frames]

# Load Emotion Assets
emotions = {}
for em in ["neutral", "happy", "thinking", "speaking"]:
    path = os.path.join(BASE_DIR, f"{em}.png")
    if os.path.exists(path):
        # Load as-is (transparency is already processed)
        img = pygame.image.load(path).convert_alpha()
        emotions[em] = img
    else:
        print(f"Warning: {path} not found.")

current_emotion = "neutral"
emotion_rotation = 0



# Load ARk Image
ark_image_path = os.path.join(BASE_DIR, 'ARk.jpeg')
ark_image_raw = pygame.image.load(ark_image_path).convert()
ark_image = pygame.transform.scale(ark_image_raw, (0, 0))  # Optional: resize if needed
# ARk Image position offset from center-right

ark_offset_x = 800  # move left/right (+/-)
ark_offset_y = 150     # move up/down (+/-)

ark_pos_x = screen.get_width() - ark_image.get_width() + ark_offset_x
ark_pos_y = (screen.get_height() // 2 - ark_image.get_height() // 2) + ark_offset_y

# Load Discord Icon
discord_icon_path = os.path.join(BASE_DIR, 'discord.png')
discord_icon_raw = pygame.image.load(discord_icon_path).convert_alpha()
discord_icon = pygame.transform.scale(discord_icon_raw, (0, 0))  # Optional: resize if needed

# Discord Icon position offset from bottom-left
discord_offset_x = 1150   # distance from left edge
discord_offset_y = 50   # distance from bottom edge

discord_pos_x = discord_offset_x
discord_pos_y = screen.get_height() - discord_icon.get_height() - discord_offset_y

# PyAudio setup
p = pyaudio.PyAudio()
# Note: LiveVoiceAgent also initializes PyAudio. This might cause conflict if not careful.
# But here we are just opening a stream.
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=512)

def get_volume(data):
    count = len(data) // 2
    format = "%dh" % count
    shorts = struct.unpack(format, data)
    sum_squares = sum(s**2 for s in shorts)
    return (sum_squares / count)**0.5



def render_calendar(surface, x, y):
    now = datetime.datetime.now()
    year = now.year
    month = now.month
    today = now.day
    
    # Use python calendar module
    cal = calendar.monthcalendar(year, month)
    
    # Header: Month Year
    header_text = now.strftime("%B %Y")
    header_surface = calendar_font.render(header_text, True, CYAN)
    surface.blit(header_surface, (x, y))
    
    y_offset = y + header_surface.get_height() + 10
    
    # Weekdays
    weekdays = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
    cell_width = 35
    margin_left = 10
    cell_height = calendar_font.get_height() + 6
    
    cur_x = x
    for idx, day in enumerate(weekdays):
        day_surface = calendar_font.render(day, True, CYAN)
        if idx > 0:
            cur_x += margin_left
        surface.blit(day_surface, (cur_x, y_offset))
        cur_x += cell_width
        
    y_offset += cell_height + 10
    
    # Days
    for week in cal:
        cur_x = x
        for i, day in enumerate(week):
            if i > 0:
                cur_x += margin_left
                
            if day == 0:
                day_str = " "
            else:
                day_str = str(day)
                
            if day == today:
                 highlight_surf = pygame.Surface((cell_width, cell_height), pygame.SRCALPHA)
                 pygame.draw.ellipse(highlight_surf, CYAN + (HIGHLIGHT_ALPHA,), highlight_surf.get_rect())
                 surface.blit(highlight_surf, (cur_x, y_offset))
                 day_surface = calendar_font.render(day_str, True, BLACK)
            else:
                 day_surface = calendar_font.render(day_str, True, CYAN)
            
            day_rect = day_surface.get_rect()
            day_pos_x = cur_x + (cell_width - day_rect.width) // 2
            day_pos_y = y_offset + (cell_height - day_rect.height) // 2
            surface.blit(day_surface, (day_pos_x, day_pos_y))
            
            cur_x += cell_width
        
        y_offset += cell_height + 6

def toggle_fullscreen(screen, fullscreen):
    if fullscreen:
        pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    else:
        pygame.display.set_mode((800, 600))
    return not fullscreen

def hand_tracking_thread():
    global hand_landmarks_global, hand_closed_global, wrist_screen_pos

    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
        cap = cv2.VideoCapture(0)  # Changed to 0 for Windows default
    except Exception as e:
        print(f"Hand tracking initialization failed: {e}")
        return

    while True:
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)

        if ENABLE_HAND_TRACKING:
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                hand_landmarks_global = hand

                # Determine if hand is closed (fingertips below lower joints)
                tips = [8, 12, 16, 20]
                closed = all(hand.landmark[tip].y > hand.landmark[tip - 2].y for tip in tips)
                hand_closed_global = closed

                # Convert wrist landmark to screen coordinates
                wrist = hand.landmark[0]
                wrist_screen_pos = (int(wrist.x * screen.get_width()), int(wrist.y * screen.get_height()))
            else:
                results = None
                hand_landmarks_global = None
                hand_closed_global = False

def run_agent(ui_queue):
    """Run the LiveVoiceAgent in a separate thread."""
    agent = LiveVoiceAgent(video_mode="none")
    agent.ui_queue = ui_queue
    # We use a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(agent.run())
    except Exception as e:
        ui_queue.put(("status", f"Agent Error: {e}"))
    finally:
        loop.close()

def main():
    global track_font, grab_active
    
    running = True
    fullscreen = False
    
    # Animation vars
    frame_idx = 0
    gif_scale = 1.0
    gif_width, gif_height = frame_surfaces[0].get_size()
    clock = pygame.time.Clock()

    # Emotion State
    current_emotion = "neutral"
    emotion_rotation = 0
    
    track_font = pygame.font.Font(font_path, 26)  # Adjust size here
    
    # Threading
    request_thread = threading.Thread(target=hand_tracking_thread, daemon=True)
    request_thread.start()
    
    # Agent Integration
    ui_queue = queue.Queue()
    agent_thread = threading.Thread(target=run_agent, args=(ui_queue,), daemon=True)
    agent_thread.start()
    
    status_text = "Initializing AI..."
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    fullscreen = toggle_fullscreen(screen, fullscreen)
                elif event.key == pygame.K_ESCAPE:
                    running = False

        try:
            # Consume UI Queue
            while not ui_queue.empty():
                try:
                    msg_type, msg_content = ui_queue.get_nowait()
                    if msg_type == "status":
                        status_text = f"[{msg_content}]"
                    elif msg_type == "text":
                        # Truncate text if too long
                        if len(msg_content) > 50:
                            status_text = msg_content[:50] + "..."
                        else:
                            status_text = msg_content
                    elif msg_type == "emotion":
                        current_emotion = msg_content
                except queue.Empty:
                    break

            # Audio visualization logic
            try:
                audio_data = stream.read(2048, exception_on_overflow=False)
                volume = get_volume(audio_data)
                scale_factor = 1 + min(volume / 1000, 1)
                gif_scale = 0.9 * gif_scale + 0.1 * scale_factor
            except Exception:
                gif_scale = 1.0

            scaled_width = int(gif_width * gif_scale)
            scaled_height = int(gif_height * gif_scale)

            # JARVIS frame (Always render)
            jarvis_frame = frame_surfaces[frame_idx]
            jarvis_scaled = pygame.transform.scale(jarvis_frame, (scaled_width, scaled_height)).convert_alpha()

            screen.fill((0, 0, 0))

            # Overlay JARVIS (Center)
            jarvis_rect = jarvis_scaled.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
            screen.blit(jarvis_scaled, jarvis_rect)

            # Draw Emotion Icon (Small, Below Face)
            if current_emotion in emotions:
                em_img = emotions[current_emotion]
                # Scale down (e.g. 50x50 or slightly larger)
                em_size = (60, 60)
                em_scaled = pygame.transform.scale(em_img, em_size)
                
                # Rotate neutral/thinking slightly for effect
                if current_emotion in ["neutral", "thinking"]:
                    emotion_rotation = (emotion_rotation + 2) % 360
                    em_scaled = pygame.transform.rotate(em_scaled, emotion_rotation)

                # Position strictly under the JARVIS face
                # jarvis_rect.bottom gives the y-coordinate of the bottom edge
                em_y = jarvis_rect.bottom + 30 
                em_rect = em_scaled.get_rect(center=(screen.get_width() // 2, em_y))
                screen.blit(em_scaled, em_rect)

            # Draw ARk image first so it stays behind Jarvis GIF
            screen.blit(ark_image, (ark_pos_x, ark_pos_y))

            # Draw Discord icon (beneath Jarvis GIF)
            screen.blit(discord_icon, (discord_pos_x, discord_pos_y))



    

            # Time
            now = datetime.datetime.now()
            current_time = now.strftime("%I:%M:%S %p")
            time_surface = clock_font.render(current_time, True, CYAN)
            time_rect = time_surface.get_rect(center=(screen.get_width() // 2, 100))
            screen.blit(time_surface, time_rect)

            # Calendar
            calendar_margin_right = 40
            calendar_cell_width = 35
            calendar_margin_left = 10
            days_in_week = 7
            calendar_width = days_in_week * calendar_cell_width + (days_in_week - 1) * calendar_margin_left
            calendar_x = screen.get_width() - calendar_width - calendar_margin_right
            render_calendar(screen, calendar_x, 60)

            # --- Draw Status (formerly Track) (bottom left) ---
            if status_text:
                track_surface = track_font.render(status_text, True, CYAN)
                track_pos = (20, screen.get_height() - track_surface.get_height() - 20) 
                screen.blit(track_surface, track_pos)


            # --- Draw To-Do List (top-left corner) ---
            todo_tasks = load_todo_tasks()
            todo_x, todo_y = 40, 300 
            todo_spacing = 28 

            for i, task in enumerate(todo_tasks[:8]):  # Display up to 8 items
                bullet = f"{task}"
                todo_surface = todo_font.render(bullet, True, CYAN)
                screen.blit(todo_surface, (todo_x, todo_y + i * todo_spacing))



            if hand_landmarks_global:
                # Draw landmarks circles
                for landmark in hand_landmarks_global.landmark:
                    x = int(landmark.x * screen.get_width())
                    y = int(landmark.y * screen.get_height())
                    pygame.draw.circle(screen, CYAN, (x, y), 6)

                # Draw connections
                connections = mp.solutions.hands.HAND_CONNECTIONS
                for connection in connections:
                    start_idx, end_idx = connection
                    start = hand_landmarks_global.landmark[start_idx]
                    end = hand_landmarks_global.landmark[end_idx]
                    start_pos = (int(start.x * screen.get_width()), int(start.y * screen.get_height()))
                    end_pos = (int(end.x * screen.get_width()), int(end.y * screen.get_height()))
                    pygame.draw.line(screen, CYAN, start_pos, end_pos, 3)



            pygame.display.flip()
            frame_idx = (frame_idx + 1) % len(frame_surfaces)
            clock.tick(30)

        except IOError as e:
            # print(f"Audio buffer overflowed: {e}")
            pass
        except Exception as e:
            print(f"Unexpected error: {e}")

    stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.quit()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
