class Buffer:
    def __init__(self, filename=None, text=""):
        self.filename = filename
        if text:
            self.lines = text.split('\n')
        else:
            self.lines = [""]
        self.cursor_y = 0
        self.cursor_x = 0

    def insert_char(self, char):
        line = self.lines[self.cursor_y]
        self.lines[self.cursor_y] = line[:self.cursor_x] + char + line[self.cursor_x:]
        self.cursor_x += 1

    def insert_newline(self):
        line = self.lines[self.cursor_y]
        new_line_content = line[self.cursor_x:]
        self.lines[self.cursor_y] = line[:self.cursor_x]
        self.lines.insert(self.cursor_y + 1, new_line_content)
        self.cursor_y += 1
        self.cursor_x = 0

    def delete_char(self):
        if self.cursor_x > 0:
            line = self.lines[self.cursor_y]
            self.lines[self.cursor_y] = line[:self.cursor_x - 1] + line[self.cursor_x:]
            self.cursor_x -= 1
        elif self.cursor_y > 0:
            # Join with previous line
            line_content = self.lines.pop(self.cursor_y)
            prev_line_len = len(self.lines[self.cursor_y - 1])
            self.lines[self.cursor_y - 1] += line_content
            self.cursor_y -= 1
            self.cursor_x = prev_line_len

    def move_cursor(self, dy, dx):
        self.cursor_y += dy
        self.cursor_x += dx

        # Clamp Y
        if self.cursor_y < 0:
            self.cursor_y = 0
        if self.cursor_y >= len(self.lines):
            self.cursor_y = len(self.lines) - 1

        # Clamp X
        line_len = len(self.lines[self.cursor_y])
        if self.cursor_x < 0:
            self.cursor_x = 0
        if self.cursor_x > line_len:
            self.cursor_x = line_len
