import pygame
import pyaudio
import struct
from PIL import Image, ImageSequence
import datetime
import subprocess
import threading
import os
import cv2
import mediapipe as mp
import numpy as np

# This code was made for MacOS, but it can be adapted to other platforms with some changes.
# Uses Python 3.10.9

ENABLE_HAND_TRACKING = True  # Set to True or False for hand control


hand_landmarks_global = None
hand_closed_global = False
wrist_screen_pos = (0, 0)
pico_x = None
pico_y = None
grab_active = False


# Initialize pygame
pygame.init()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load custom font (Orbitron)
font_path = os.path.join(BASE_DIR, 'Orbitron-VariableFont_wght.ttf')
clock_font = pygame.font.Font(font_path, 80)
calendar_font = pygame.font.Font(font_path, 20)
description_font = pygame.font.Font(font_path, 16)
pico_description_lines = [
    "MateoTechLab File: Project Pico",
    "Personality is stable, but can be customized",
    "Uses an ESP32 Wrover module",
    "Open-source and community-driven",
    "Supports various sensors and modules",
    "Ideal for personal projects and educational purposes",
] # This can be customized to your liking... or you can keep Pico :D

track_font = pygame.font.SysFont("SF Mono", 18)



CYAN = (0, 255, 255)
BLACK = (0, 0, 0)
HIGHLIGHT_ALPHA = 80

todo_file_path = os.path.join(BASE_DIR, 'todo.txt') # Make a file called .todo.txt in the same directory as this script and write your to do list inside
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
pygame.display.set_caption('J.A.R.V.I.S')

# Load JARVIS face GIF
gif_path = os.path.join(BASE_DIR, 'jarvis.gif')
gif = Image.open(gif_path)
frames = [frame.copy().convert("RGBA") for frame in ImageSequence.Iterator(gif)]
frame_surfaces = [pygame.image.frombuffer(frame.tobytes(), frame.size, "RGBA") for frame in frames]

# Load Pico GIF
pico_gif_path = os.path.join(BASE_DIR, 'picogram.gif')
pico_gif = Image.open(pico_gif_path)
pico_frames = [frame.copy().convert("RGBA") for frame in ImageSequence.Iterator(pico_gif)]
pico_surfaces = [pygame.image.frombuffer(f.tobytes(), f.size, "RGBA") for f in pico_frames]

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

# Load record icon
record_icon_path = os.path.join(BASE_DIR, 'record.png')
record_icon = pygame.image.load(record_icon_path).convert_alpha()

# Resize the image (optional)
record_icon_size = (100, 100)  # Change this if needed
record_icon = pygame.transform.scale(record_icon, record_icon_size)


# Load sun phase icons
sunrise_icon = pygame.image.load(os.path.join(BASE_DIR, 'sunrise.png')).convert_alpha()
sun_icon = pygame.image.load(os.path.join(BASE_DIR, 'sun.png')).convert_alpha()
sunset_icon = pygame.image.load(os.path.join(BASE_DIR, 'sunset.png')).convert_alpha()

# Resize all icons to same size
sun_icon_size = (350, 350)  # adjust as needed
sunrise_icon = pygame.transform.scale(sunrise_icon, sun_icon_size)
sun_icon = pygame.transform.scale(sun_icon, sun_icon_size)
sunset_icon = pygame.transform.scale(sunset_icon, sun_icon_size)


# PyAudio setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=512)

def get_volume(data):
    count = len(data) // 2
    format = "%dh" % count
    shorts = struct.unpack(format, data)
    sum_squares = sum(s**2 for s in shorts)
    return (sum_squares / count)**0.5

def get_calendar_data():
    try:
        output = subprocess.check_output(['cal', '-h'], text=True)
        lines = output.strip().split('\n')
        return lines
    except Exception as e:
        print(f"Calendar fetch error: {e}")
        return []

def get_sun_icon():
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 11:
        return sunrise_icon  # Morning
    elif 11 <= current_hour < 17:
        return sun_icon      # Midday
    else:
        return sunset_icon   # Evening/Night

def render_calendar(surface, x, y):
    lines = get_calendar_data()
    if not lines:
        return

    today = datetime.datetime.now().day
    weekdays = lines[1].split()
    cell_width = 35
    margin_left = 10
    cell_height = calendar_font.get_height() + 6

    header_surface = calendar_font.render(lines[0], True, CYAN)
    surface.blit(header_surface, (x, y))
    y_offset = y + header_surface.get_height() + 10

    cur_x = x
    for idx, day in enumerate(weekdays):
        day_surface = calendar_font.render(day, True, CYAN)
        if idx > 0:
            cur_x += margin_left
        surface.blit(day_surface, (cur_x, y_offset))
        cur_x += cell_width

    y_offset += cell_height + 10

    for week_line in lines[2:]:
        cur_x = x
        for i in range(7):
            start = i * 3
            day_str = week_line[start:start+3].strip()
            if day_str == '':
                day_str = ' '

            if i > 0:
                cur_x += margin_left

            if day_str.isdigit() and int(day_str) == today:
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

track = ""
track_lock = threading.Lock()

def fetch_track():
    global track
    try:
        running = subprocess.check_output(
            'ps -ef | grep "MacOS/Spotify" | grep -v "grep" | wc -l',
            shell=True, text=True
        ).strip()
        if running == "0":
            new_track = ""
        else:
            new_track = subprocess.check_output(
                """osascript -e 'tell application "Spotify"
                    set t to current track
                    return artist of t & " - " & name of t
                end tell'""",
                shell=True, text=True
            ).strip()
        with track_lock:
            track = new_track
    except:
        with track_lock:
            track = ""

def hand_tracking_thread():
    global hand_landmarks_global, hand_closed_global, wrist_screen_pos

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    cap = cv2.VideoCapture(1)  # MacBook's camera

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

def main():
    global track_font  # So you can keep the correct font
    running = True
    fullscreen = False
    frame_idx = 0
    pico_idx = 0
    gif_scale = 1.0
    gif_width, gif_height = frame_surfaces[0].get_size()
    clock = pygame.time.Clock()
    track_font = pygame.font.Font(font_path, 26)  # Adjust size here
    track_update_ms = 3000
    last_track_ms = 0
    threading.Thread(target=hand_tracking_thread, daemon=True).start()
    global pico_x, pico_y, grab_active


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
            audio_data = stream.read(2048, exception_on_overflow=False)
            volume = get_volume(audio_data)

            scale_factor = 1 + min(volume / 1000, 1)
            gif_scale = 0.9 * gif_scale + 0.1 * scale_factor

            scaled_width = int(gif_width * gif_scale)
            scaled_height = int(gif_height * gif_scale)

            now_ms = pygame.time.get_ticks()
            if now_ms - last_track_ms >= track_update_ms:
                threading.Thread(target=fetch_track, daemon=True).start()
                last_track_ms = now_ms



            # JARVIS frame
            jarvis_frame = frame_surfaces[frame_idx]
            jarvis_scaled = pygame.transform.scale(jarvis_frame, (scaled_width, scaled_height)).convert_alpha()
            jarvis_tint = pygame.Surface((scaled_width, scaled_height), pygame.SRCALPHA)
            jarvis_tint.fill(CYAN + (255,))
            jarvis_scaled.blit(jarvis_tint, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

            screen.fill((0, 0, 0))

            # Overlay JARVIS
            jarvis_rect = jarvis_scaled.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
            screen.blit(jarvis_scaled, jarvis_rect)

            # Draw ARk image first so it stays behind Jarvis GIF
            screen.blit(ark_image, (ark_pos_x, ark_pos_y))

            # Draw Discord icon (beneath Jarvis GIF)
            screen.blit(discord_icon, (discord_pos_x, discord_pos_y))

            # PICO frame
            pico_frame = pico_surfaces[pico_idx]
            pico_target_width = 600
            pico_scale = pico_target_width / pico_frame.get_width()
            pico_target_height = int(pico_frame.get_height() * pico_scale)
            pico_scaled = pygame.transform.scale(pico_frame, (pico_target_width, pico_target_height)).convert_alpha()

            pico_tint = pygame.Surface((pico_target_width, pico_target_height), pygame.SRCALPHA)
            pico_tint.fill(CYAN + (255,))
            pico_scaled.blit(pico_tint, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

            # Initialize Pico position once
            if pico_x is None or pico_y is None:
                pico_x = screen.get_width() - pico_target_width - -850 
                pico_y = screen.get_height() - pico_target_height - -170 


            # Get middle finger MCP position if available
            if hand_landmarks_global:
                mcp = hand_landmarks_global.landmark[9]  # middle finger MCP
                mcp_screen_pos = (int(mcp.x * screen.get_width()), int(mcp.y * screen.get_height()))
            else:
                mcp_screen_pos = None

            pico_rect = pygame.Rect(pico_x, pico_y, pico_target_width, pico_target_height)

            if hand_landmarks_global and hand_closed_global and mcp_screen_pos:
                # Only start grabbing if middle MCP is over Pico
                if pico_rect.collidepoint(mcp_screen_pos):
                    grab_active = True

            if grab_active:
                if hand_closed_global and mcp_screen_pos:
                    # Move Pico centered on middle MCP position
                    pico_x = mcp_screen_pos[0] - pico_target_width // 2
                    pico_y = mcp_screen_pos[1] - pico_target_height // 2
                else:
                    grab_active = False


            pico_pos = (int(pico_x), int(pico_y))
            screen.blit(pico_scaled, pico_pos)


            # Multi-line Pico Description (left-aligned)
            line_spacing = 6
            text_margin = 80  # space between the gif edge and the text

            for i, line in enumerate(pico_description_lines):
                line_surface = description_font.render(line, True, CYAN)
                line_x = pico_pos[0] + text_margin  # align left with margin
                line_y = pico_pos[1] + pico_target_height + -30 + i * (line_surface.get_height() + line_spacing)
                screen.blit(line_surface, (line_x, line_y))

    
                # Position at bottom left
            record_x = 30 #
            record_y = screen.get_height() - record_icon_size[1] - 60  # 30px from bottom

            # Draw record icon
            screen.blit(record_icon, (record_x, record_y))
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

# --- Draw current track (bottom left) ---
            with track_lock:
                current_track = track

            if current_track:
                track_surface = track_font.render(current_track, True, CYAN)
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

            # Get the current sun icon
            current_sun_icon = get_sun_icon()

            # Position at top-left
            sun_pos = (30, -30) 
            screen.blit(current_sun_icon, sun_pos)

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
            pico_idx = (pico_idx + 1) % len(pico_surfaces)
            clock.tick(30)

        except IOError as e:
            print(f"Audio buffer overflowed: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.quit()

if __name__ == '__main__':
    main()