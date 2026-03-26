# -------------------------------------------------------------
# Tools
# -------------------------------------------------------------
import subprocess
from pathlib import Path
from inspect import cleandoc, getdoc


def default_tools() -> list:
    return [read_file, write_file, bash] # , inline_image]

def tool_prompt(tools) -> str:
    tool_names = "## Tools\n\n" + "\n".join([f"- {tool.__name__}" for tool in tools]) 
    return tool_names if tools else "## Tools are NOT available"

def read_file(path: str) -> str:
    """
    Read the contents of a file

    Args:
        path (str): Path to the file to read

    Returns:
        str: the content of the file to read or an error message
    """
    print(f"    read_file: {path}")
    try:
        return Path(path).read_text()
    except:
        return "Error: Could not read file"

def write_file(path: str, content: str) -> str:
    """
    Write content to file, intermediate directories automatically added if necessary.

    Args:
        path (str): Path to the file to write
        content (str): content to write

    Returns:
        str: status message
    """
    print(f"    write_file: {path}")
    try:
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
        return f"Wrote content to {path}"
    except:
        return "Error: Could not write file"

def bash(command: str) -> str:
    """
    Execute a bash command and return result

    Args:
        command (str): The bash command to execute

    Returns:
        str: the result of the command or an error message
    """
    print(f"    bash: {command[:50]}{'...' if len(command) >= 50 else ''}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        output = result.stdout
        return output or "Command executed successfully (no output)"
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out"
        
# -------------------------------------------------------------
# Base Agent
# -------------------------------------------------------------
import ollama

class Agent:
    """A simple AI agent that can answer questions by planning and performing multiple steps"""

    MAX_CTX_SIZE = 65536
    
    def __init__(self, host: str, model: str, tools: list | None = None):
        """
        Instantiate an agent
        Provide a host, model name, and (optionally) a list of tools
        """
        self.model = model
        self.tools = tools or []
        self.known_tools = {tool.__name__: tool for tool in self.tools}
        self.client = ollama.Client(host=host, timeout=300)
        self.ctx_size = min(self.get_model_ctx_size(), self.MAX_CTX_SIZE)
        
        system_message = self.generate_system_prompt()
        self.messages = [{"role":"system", "content": system_message}]

    def get_model_ctx_size(self):
        FALLBACK_CTX_SIZE = 4096
        info = self.client.show(self.model).modelinfo
        res = [val for key, val in info.items() if key.endswith("context_length")] + [FALLBACK_CTX_SIZE]
        return res[0]

    def context_usage(self):
        # Estimate how much context (in tokens) we are currently using
        # Guesstimate: 1 token ≈ 4 characters (including spaces).
        return int(sum([len(msg.get("content", "")) for msg in self.messages])/4)
        
    def generate_system_prompt(self) -> str:
        preamble = self.system_prompt_preamble()
        tools = tool_prompt(self.tools)
        return f"{preamble}\n\n{tools}"

    def system_prompt_preamble(self):
        prompt = """
            You are an AI assistant with access to tools, capable of writing code and issuing commands if the toolset allows.
            Your response will be rendered as markdown, you may use image links, e.g. ![](work/foo.png) to show images if needed.
            ALWAYS use the local './work/' directory for intermediate files as you don't have access to anything under '/', e.g. ''/tmp' etc.
            """
        return cleandoc(prompt)
        
    def task(self, user_query: str, max_steps: int = 16):
        """Public interface"""                        
        result = self._perform_steps(user_query, max_steps)
        ## Pretty-print output
        from IPython.display import display, Markdown
        display(Markdown("---\n\n" + result + "\n\n---\n"))

    def _perform_steps(self, step_input: str, max_steps: int):
        i = 0

        # Add user message
        self.messages.append({"role": "user", "content": step_input})

        while i < max_steps:
            i += 1
            print(f"Step #{i}, please wait ...")
            print(f"Context use estimate: {self.context_usage()} of {self.ctx_size} tokens")
                
            response = self._chat(messages=self.messages, tools=self.tools)
            message = response.get("message", {})
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
            
            # Add assistant message to history
            self.messages.append(message)
            
            # If no tool calls, we're done. But if neither content nor tools we have an error...
            if not tool_calls:
                if content:
                    return content
                else:                
                    return "Error: Neither content nor tool calls."
            
            # If there's content **and** tool calls, display content (shouldn't really happen...)
            if content:
                print(f"--- Assistant (internal) ------\n{content}")

            # Execute tool calls
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                tool_name = function.get("name", "")
                arguments = function.get("arguments", {})

                # Display tool call
                print(f"--- Tool Call: {tool_name}")

                # Execute the tool
                try:
                    tool = self.known_tools[tool_name]
                except Exception:
                    self.messages.append({"role": "tool", "content": f"Error: Unknown tool '{tool_name}'"})
                    break
                try:
                    result = tool(**arguments)
                except Exception as e:
                    self.messages.append({"role": "tool", "content": f"Error: {e}"})
                    break
                
                # Add tool result to messages
                self.messages.append({"role": "tool", "content": result})

        # We hit the maximum number of steps, the LLM is likely very confused
        return f"Agent was unable to answer your question in the maximal number of steps ({max_steps})"

    def _chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict | list[dict]:
        """Send a chat request to Ollama."""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": self.ctx_size}
            }

            if tools:
                kwargs["tools"] = tools
                
            return self.client.chat(**kwargs)
            
        except Exception as e:
            return {"message": {"role": "assistant", "content": f"Error communicating with Ollama: {e}"}}
 
# -------------------------------------------------------------
# Helper
# -------------------------------------------------------------
import json
from IPython import display

def show_messages(messages):
    """Helper function to view messages in a nice format"""
    s = json.loads(json.dumps(messages, default=dict))
    return display.JSON(s, root="Message history")

# -------------------------------------------------------------
# Skill loading
# -------------------------------------------------------------
import os
import glob
from inspect import cleandoc

import frontmatter


def load_skills(skill_dir: str) -> dict:

    skills = {}
    
    files = glob.glob(f"{skill_dir}/*/SKILL.md")
    for file in files:
        metadata = frontmatter.load(file).metadata        
        if "name" in metadata and "description" in metadata:
            skills[metadata["name"]] = metadata["description"]
        
    return skills
    

def skill_prompt(skills: dict, skill_dir: str) -> str:
        
    if not skills:
        return "\n\n## Skills\n\nNo skills available."
    
    skill_descriptions = "\n".join([f"- {name}: {desc}" for name, desc in skills.items()])

    prompt = f"""
## Skills

Skills are specialized capabilities stored in the '{skill_dir}' directory. 
When a user asks you to do something, FIRST check if any available skill decription matches their request. If so, use that skill WITHOUT being explicitly told.

**Example:**

- User: "List phone deals" → Automatically use 'watching-deals' skill

### Available skills (name: description)

{skill_descriptions}

### How to Use Skills

**Step-by-step instructions:**

1. Use 'read_file' tool to read skill definition from '{skill_dir}/{{name}}/SKILL.md', e.g. read_file('{skill_dir}/watching-deals/SKILL.md')
2. Follow the detailed step-by-step instructions in the skill file
3. The skill may reference supporting files (scripts, benchmarks, templates) - read and use those as needed
4. Skills can invoke other skills (composability)
5. Never make up analysis - use the skill's methods and data

"""
    
    return prompt

# -------------------------------------------------------------
# SkilledAgent
# -------------------------------------------------------------
class SkilledAgent(Agent):

    def __init__(self, host: str, model: str, skill_dir: str):
        self.skill_dir = skill_dir
        self.skills = load_skills(self.skill_dir)
        super().__init__(host, model, default_tools())
        
    def generate_system_prompt(self) -> str:
        base = super().generate_system_prompt()
        skill_addition = skill_prompt(self.skills, self.skill_dir)
        return f"{base}\n\n{skill_addition}"


