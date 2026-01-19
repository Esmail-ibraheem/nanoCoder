from pathlib import Path
from typing import Any, Dict, List, Optional
from rich.console import Console, Group
from rich.theme import Theme
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.live import Live
from rich import box
import re
import time

from config.config import Config
from tools.base import ToolConfirmation
from utils.paths import display_path_rel_to_cwd
from utils.text import truncate_text

# nanoCoder Agent Colors
COLORS = {
    "primary": "#FFB800",      # Golden yellow
    "secondary": "#00D9FF",    # Cyan
    "success": "#00FF88",      # Green
    "error": "#FF4444",        # Red
    "warning": "#FFB800",      # Orange
    "info": "#00D9FF",         # Cyan
    "muted": "#666666",        # Gray
    "text": "#FFFFFF",         # White
}

AGENT_THEME = Theme({
    "info": COLORS["info"],
    "warning": COLORS["warning"],
    "error": COLORS["error"],
    "success": COLORS["success"],
    "muted": COLORS["muted"],
    "user": COLORS["secondary"] + " bold",
    "assistant": COLORS["primary"],
    "tool": COLORS["secondary"] + " bold",
    "highlight": COLORS["primary"] + " bold",
})

def get_console() -> Console:
    return Console(theme=AGENT_THEME, highlight=False)

class TUI:
    def __init__(self, config: Config, console: Console | None = None) -> None:
        self.console = console or get_console()
        self.config = config
        self.cwd = self.config.cwd
        self._max_block_tokens = 2500
        self._live: Optional[Live] = None
        self._current_content = ""
        self._tool_args_by_call_id: Dict[str, Dict[str, Any]] = {}

    def print_banner(self):
        banner = r"""
[bold #FFB800]
║                                                                                                 
║   ███╗   ██╗ █████╗ ███╗   ██╗ ██████╗  ██████╗ ██████╗ ██████╗ ███████╗██████╗              
║   ████╗  ██║██╔══██╗████╗  ██║██╔═══██╗██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗             
║   ██╔██╗ ██║███████║██╔██╗ ██║██║   ██║██║     ██║   ██║██║  ██║█████╗  ██████╔╝             
║   ██║╚██╗██║██╔══██║██║╚██╗██║██║   ██║██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗             
║   ██║ ╚████║██║  ██║██║ ╚████║╚██████╔╝╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║             
║   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝             
║                                                                                                  
║                          [#00D9FF]n a n o C O D E R[/#00D9FF]                                     
║                                                                                                  
║            [dim]Your specialized AI assistant for deep coding tasks[/dim]                                  
[/bold #FFB800]
"""
        self.console.print(banner)

    def print_welcome(self, title: str, lines: list[str]) -> None:
        self.print_banner()
        info_text = f"[bold #00D9FF]{title}[/bold #00D9FF]\n\n"
        for line in lines:
            parts = line.split(": ", 1)
            if len(parts) == 2:
                info_text += f"[#FFB800]{parts[0]}:[/#FFB800] {parts[1]}\n"
            else:
                info_text += f"{line}\n"
        
        self.console.print(Panel(
            info_text.strip(),
            border_style="#00D9FF",
            box=box.DOUBLE,
            padding=(1, 2)
        ))
        self.console.print(f"[dim #666666]{'─' * 60}[/dim #666666]")

    def begin_assistant(self) -> None:
        self._current_content = ""
        self._live = Live(
            Panel("", title="[#FFB800]Assistant[/#FFB800]", border_style="#FFB800", box=box.ROUNDED, padding=(1, 2)),
            console=self.console,
            refresh_per_second=10,
            transient=True
        )
        self._live.start()

    def stream_assistant_delta(self, content: str) -> None:
        self._current_content += content
        if self._live:
            self._live.update(Panel(
                self._current_content,
                title="[#FFB800]Assistant[/#FFB800]",
                border_style="#FFB800",
                box=box.ROUNDED,
                padding=(1, 2)
            ))

    def end_assistant(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None
        # Print final version statically
        self.console.print(Panel(
            self._current_content,
            title="[#FFB800]Assistant[/#FFB800]",
            border_style="#FFB800",
            box=box.ROUNDED,
            padding=(1, 2)
        ))

    def tool_call_start(self, call_id: str, name: str, tool_kind: str | None, arguments: dict[str, Any]) -> None:
        self._tool_args_by_call_id[call_id] = arguments
        self.console.print(f"\n[bold #FFB800]🔧 Tool:[/bold #FFB800] [#00D9FF]{name}[/#00D9FF]")
        if arguments:
            table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
            table.add_column("Key", style="#FFB800")
            table.add_column("Value", style="white")
            for k, v in arguments.items():
                val = str(v)
                if k in ("path", "cwd") and self.cwd:
                    val = str(display_path_rel_to_cwd(val, self.cwd))
                value_str = val[:100] + "..." if len(val) > 100 else val
                table.add_row(k, value_str)
            self.console.print(table)

    def tool_call_complete(self, call_id: str, name: str, tool_kind: str | None, success: bool, output: str, error: str | None, metadata: dict[str, Any] | None, diff: str | None, truncated: bool, exit_code: int | None) -> None:
        status_color = "#00FF88" if success else "#FF4444"
        status_text = "Result" if success else "Error"
        
        blocks = []
        if not success and error:
            blocks.append(Text(error, style="error"))
        
        if success:
            if diff:
                blocks.append(Syntax(diff, "diff", theme="monokai", word_wrap=True))
            elif name == "read_file" and metadata:
                code = output.split("\n\n", 1)[-1] if "\n\n" in output else output
                blocks.append(Syntax(code, self._guess_language(metadata.get("path")), theme="monokai", line_numbers=True))
            else:
                blocks.append(Text(truncate_text(output, "", self._max_block_tokens)))

        panel = Panel(
            Group(*blocks) if blocks else Text("(no output)", style="muted"),
            title=f"[{status_color}]{status_text}[/{status_color}]",
            border_style=status_color,
            box=box.ROUNDED,
            padding=(1, 2)
        )
        self.console.print(panel)

    def handle_confirmation(self, confirmation: ToolConfirmation) -> bool:
        from rich.prompt import Prompt
        self.console.print(Panel(
            Text.assemble(
                (f"Approval Required: {confirmation.tool_name}\n", "warning"),
                (confirmation.description, "white")
            ),
            border_style="warning",
            box=box.ROUNDED,
            padding=(1, 2)
        ))
        resp = Prompt.ask("\nApprove? (y/n)", choices=["y", "n"], default="n")
        return resp.lower() == "y"

    def _guess_language(self, path: Optional[str]) -> str:
        if not path: return "text"
        suffix = Path(path).suffix.lower()
        return {".py": "python", ".js": "javascript", ".ts": "typescript", ".md": "markdown"}.get(suffix, "text")

    def show_help(self) -> None:
        help_text = """
# Available Commands
- `/help` - Show this help
- `/exit` / `/quit` - Exit
- `/clear` - Clear history
- `/config` - Show config
- `/model <name>` - Change model
- `/approval <mode>` - Change approval
- `/stats` - Session stats
- `/tools` - List tools
"""
        self.console.print(Markdown(help_text))
