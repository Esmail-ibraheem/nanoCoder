import curses
import os
import sys
from editor import Editor

def main(stdscr, filename=None):
    # Curses setup usually handled by wrapper, but we can tweak
    os.environ.setdefault('ESCDELAY', '25') # Decrease delay for escape key
    
    editor = Editor(stdscr, filename)
    editor.run()

if __name__ == "__main__":
    try:
        # Create a test file if it doesn't exist for easier verification
        if not os.path.exists("test.txt"):
           with open("test.txt", "w") as f:
               f.write("Hello World\nThis is a test file.\nWelcome to the Python IDE.")
        
        # Load argument or default
        filename = None
        if len(sys.argv) > 1:
            filename = sys.argv[1]
        
        curses.wrapper(main, filename)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
