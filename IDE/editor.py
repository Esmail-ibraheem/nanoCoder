import curses
import sys
import os
from buffer import Buffer
from window import Window

class Editor:
    def __init__(self, stdscr, filename=None):
        self.stdscr = stdscr
        self.mode = 'NORMAL' # NORMAL, INSERT, COMMAND
        self.buffers = []
        
        content = ""
        if filename and os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    content = f.read()
            except:
                pass
        elif filename:
             # File doesn't exist, will be created on save
             pass
             
        self.buffers.append(Buffer(filename, content))
        self.active_buffer_idx = 0
        
        # Determine window size
        h, w = stdscr.getmaxyx()
        self.window = Window(self.buffers[0], h - 1, w, 0, 0) # Reserve last line for status
        
        self.status_line_y = h - 1
        self.command_buffer = ""
        self.running = True

    def run(self):
        while self.running:
            self.render()
            self.handle_input()

    def render(self):
        self.stdscr.erase()
        self.window.render(self.stdscr)
        self.draw_status_bar()
        
        # Position cursor
        if self.mode == 'COMMAND':
            try:
                self.stdscr.addstr(self.status_line_y, 0, ":" + self.command_buffer)
                self.stdscr.move(self.status_line_y, 1 + len(self.command_buffer))
            except:
                pass
        else:
            y, x = self.window.get_cursor_screen_pos()
            try:
                self.stdscr.move(y, x)
            except:
                pass
            
        self.stdscr.refresh()

    def draw_status_bar(self):
        h, w = self.stdscr.getmaxyx()
        mode_str = f"-- {self.mode} --"
        status_str = f" {self.window.buffer.filename or '[No Name]'} "
        cursor_str = f" Ln {self.window.buffer.cursor_y + 1}, Col {self.window.buffer.cursor_x + 1} "
        
        # Draw background for status bar
        try:
            # Simple status bar
            self.stdscr.attron(curses.A_REVERSE)
            self.stdscr.addstr(self.status_line_y, 0, " " * (w - 1))
            self.stdscr.addstr(self.status_line_y, 0, mode_str)
            self.stdscr.addstr(self.status_line_y, len(mode_str) + 1, status_str)
            self.stdscr.addstr(self.status_line_y, w - len(cursor_str) - 1, cursor_str)
            self.stdscr.attroff(curses.A_REVERSE)
        except curses.error:
            pass

    def handle_input(self):
        try:
            key = self.stdscr.getkey()
        except:
            return

        if self.mode == 'NORMAL':
            self.handle_normal_mode(key)
        elif self.mode == 'INSERT':
            self.handle_insert_mode(key)
        elif self.mode == 'COMMAND':
            self.handle_command_mode(key)

    def handle_normal_mode(self, key):
        if key == 'q':
            # Temporary quit for debug, real quit should be :q
            pass 
        elif key == ':':
            self.mode = 'COMMAND'
            self.command_buffer = ""
        elif key == 'i':
            self.mode = 'INSERT'
        elif key == 'h':
            self.window.buffer.move_cursor(0, -1)
        elif key == 'j':
            self.window.buffer.move_cursor(1, 0)
        elif key == 'k':
            self.window.buffer.move_cursor(-1, 0)
        elif key == 'l':
            self.window.buffer.move_cursor(0, 1)

    def handle_insert_mode(self, key):
        if key == 'KEY_c(91)' or key == '\x1b': # Esc (often sends escape sequence) or check specific ESC key
             # Note: Curses getkey definition of ESC can vary. 
             # Usually standard ASCII 27 is ESC.
             # For now, let's assume 'ESC' string or char 27.
             pass
        
        if len(key) == 1 and ord(key) == 27: # ESC
            self.mode = 'NORMAL'
            # Move cursor back one step when exiting insert mode (vim style)
            self.window.buffer.move_cursor(0, -1)
            return

        if key == "KEY_BACKSPACE" or key == '\b' or key == '\x7f':
            self.window.buffer.delete_char()
        elif key == '\n' or key == '\r':
            self.window.buffer.insert_newline()
        elif len(key) == 1:
            self.window.buffer.insert_char(key)

    def handle_command_mode(self, key):
        if len(key) == 1 and ord(key) == 27: # ESC
            self.mode = 'NORMAL'
            return
            
        if key == '\n' or key == '\r':
            self.execute_command()
            self.mode = 'NORMAL'
        elif key == "KEY_BACKSPACE" or key == '\b' or key == '\x7f':
            if len(self.command_buffer) > 0:
                self.command_buffer = self.command_buffer[:-1]
            else:
                self.mode = 'NORMAL'
        elif len(key) == 1:
            self.command_buffer += key

    def execute_command(self):
        cmd = self.command_buffer.strip()
        if cmd == 'q':
            self.running = False
        elif cmd.startswith('w'):
            parts = cmd.split()
            if len(parts) > 1:
                filename = parts[1]
                self.save_buffer(filename)
            elif self.window.buffer.filename:
                self.save_buffer(self.window.buffer.filename)
            else:
                # Error: no filename
                pass
    
    def save_buffer(self, filename):
        try:
            with open(filename, 'w') as f:
                f.write('\n'.join(self.window.buffer.lines))
            self.window.buffer.filename = filename
        except Exception as e:
            # Show error?
            pass
