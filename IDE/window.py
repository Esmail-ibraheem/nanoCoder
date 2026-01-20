import curses

class Window:
    def __init__(self, buffer, height, width, y, x):
        self.buffer = buffer
        self.height = height
        self.width = width
        self.y = y
        self.x = x
        self.scroll_offset_y = 0
        self.scroll_offset_x = 0

    def render(self, stdscr):
        # Update scroll to keep cursor in view
        self._update_scroll()

        for i in range(self.height):
            buf_row = self.scroll_offset_y + i
            if buf_row >= len(self.buffer.lines):
                # Draw tildes for empty lines past end of buffer (Vim style)
                if i < self.height - 1: # Avoid drawing on last line if it causes issues
                   try:
                       stdscr.addstr(self.y + i, self.x, "~")
                   except curses.error:
                       pass
                continue

            line = self.buffer.lines[buf_row]
            # Handle horizontal scrolling
            render_line = line[self.scroll_offset_x:]
            
            # Truncate to width
            if len(render_line) > self.width:
                render_line = render_line[:self.width]

            try:
                stdscr.addstr(self.y + i, self.x, render_line)
            except curses.error:
                pass # Ignore errors from drawing to bottom-right corner

        # Draw status line (simplified)
        status = f" {self.buffer.filename or '[No Name]'} - ({self.buffer.cursor_y+1},{self.buffer.cursor_x+1}) "
        try:
             # Just basic visual aid, renders at the very bottom of the assigned area if there's room, 
             # or we rely on the main loop to draw a status bar outside the window.
             # For now, let's leave the window purely for text.
             pass
        except:
             pass

    def _update_scroll(self):
        # Vertical Scroll
        if self.buffer.cursor_y < self.scroll_offset_y:
            self.scroll_offset_y = self.buffer.cursor_y
        elif self.buffer.cursor_y >= self.scroll_offset_y + self.height:
            self.scroll_offset_y = self.buffer.cursor_y - self.height + 1

        # Horizontal Scroll
        if self.buffer.cursor_x < self.scroll_offset_x:
            self.scroll_offset_x = self.buffer.cursor_x
        elif self.buffer.cursor_x >= self.scroll_offset_x + self.width:
             self.scroll_offset_x = self.buffer.cursor_x - self.width + 1

    def get_cursor_screen_pos(self):
        screen_y = self.y + (self.buffer.cursor_y - self.scroll_offset_y)
        screen_x = self.x + (self.buffer.cursor_x - self.scroll_offset_x)
        return screen_y, screen_x
