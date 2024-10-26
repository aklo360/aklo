#!/usr/bin/env python3

# Existing imports
import os
os.environ['PYTHONWARNINGS'] = 'ignore:Unverified HTTPS request'
import warnings
warnings.filterwarnings('ignore', message='.*OpenSSL.*')

import click
import yaml
import json
from datetime import datetime, timezone
import pytz
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from anthropic import Anthropic
import openai
from typing import Optional, Dict, Any, List, Tuple
import sqlite3
import time
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class MemoryManager:
    def __init__(self, db_path: str, context_window: int = 20):
        self.db_path = db_path
        self.context_window = context_window
        self.setup_database()
    
    def setup_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    role TEXT,
                    content TEXT,
                    model TEXT,
                    service TEXT
                )
            """)
            
            # Create index for faster retrieval
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memory(timestamp)")
    
    def add_interaction(self, role: str, content: str, model: str, service: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memory (timestamp, role, content, model, service) VALUES (?, ?, ?, ?, ?)",
                (time.time(), role, content, model, service)
            )
    
    def get_recent_context(self, max_chars: int = 12000) -> List[Dict[str, str]]:
        """
        Get recent context while respecting token limits
        Returns list of messages in format expected by AI models
        """
        messages = []
        total_chars = 0
        
        with sqlite3.connect(self.db_path) as conn:
            # Get recent messages, ordered by timestamp
            for row in conn.execute(
                "SELECT role, content FROM memory ORDER BY timestamp DESC LIMIT ?",
                (self.context_window,)
            ):
                message = {"role": row[0], "content": row[1]}
                message_chars = len(row[1])
                
                if total_chars + message_chars > max_chars:
                    break
                
                messages.insert(0, message)  # Insert at beginning to maintain chronological order
                total_chars += message_chars
        
        return messages

    def search_memory(self, query: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            results = []
            for row in conn.execute(
                """
                SELECT timestamp, role, content, model, service 
                FROM memory 
                WHERE content LIKE ? 
                ORDER BY timestamp DESC
                """,
                (f"%{query}%",)
            ):
                results.append({
                    'timestamp': datetime.fromtimestamp(row[0]).strftime('%Y-%m-%d %H:%M:%S'),
                    'role': row[1],
                    'content': row[2],
                    'model': row[3],
                    'service': row[4]
                })
            return results

class AITools:
    def __init__(self):
        self.config = {
            'openai': {
                'default_model': os.getenv('AI_OPENAI_DEFAULT_MODEL', 'gpt-3.5-turbo'),
                'max_tokens': int(os.getenv('AI_MAX_TOKENS', '4096'))
            },
            'anthropic': {
                'default_model': os.getenv('AI_ANTHROPIC_DEFAULT_MODEL', 'claude-3-haiku-20240307'),
                'max_tokens': int(os.getenv('AI_MAX_TOKENS', '4096'))
            },
            'default_service': os.getenv('AI_DEFAULT_SERVICE', 'anthropic'),
            'temperature': float(os.getenv('AI_DEFAULT_TEMPERATURE', '0.7')),
            'db_path': os.path.expanduser('~/.ai-tools/memory.db'),
            'context_window': int(os.getenv('AI_CONTEXT_WINDOW', '20'))  # Number of previous messages to include
        }
        
        # Ensure database directory exists
        Path(self.config['db_path']).parent.mkdir(parents=True, exist_ok=True)
        
        self.memory = MemoryManager(self.config['db_path'], self.config['context_window'])
        self.client = openai.OpenAI()

    def get_chat_response(self, prompt: str, service: str, model: Optional[str] = None,
                         temperature: Optional[float] = None, computer_control: bool = False) -> str:
        
        model = model or self.config[service]['default_model']
        temp = temperature or self.config['temperature']
        
        # Get recent context
        context = self.memory.get_recent_context()
        
        # Add current time information
        est = pytz.timezone('US/Eastern')
        current_time = datetime.now(est)
        time_info = f"Current date and time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} EST"
        
        # Prepare system message
        system_message = "You are an AI assistant with access to the current time and potentially reference files. Use this information when answering questions.\n\n" + time_info
        
        # Add current prompt to context
        context.append({"role": "user", "content": prompt})
        
        # Store user message in memory
        self.memory.add_interaction("user", prompt[:1000], model, service)  # Store only first 1000 chars to save space
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task(description="Thinking...", total=None)
            
            if service == 'anthropic':
                client = Anthropic()
                
                if computer_control:
                    progress.update(task, description="Preparing computer control...")
                    response_text = "Computer Control API is not yet available."
                else:
                    progress.update(task, description="Generating response...")
                    response = client.messages.create(
                        model=model,
                        max_tokens=self.config['anthropic']['max_tokens'],
                        temperature=temp,
                        system=system_message,
                        messages=[{"role": "user", "content": time_info + "\n\n" + prompt}]
                    )
                    response_text = str(response.content[0].text) if hasattr(response.content, '__getitem__') else str(response.content)
            
            elif service == 'openai':
                progress.update(task, description="Generating response...")
                openai_context = [{"role": "system", "content": system_message}] + context
                response = self.client.chat.completions.create(
                    model=model,
                    messages=openai_context,
                    temperature=temp
                )
                response_text = response.choices[0].message.content
        
        # Add model information to the response
        if not response_text.endswith(f"Response by {model}"):
            response_text += f"\n\nResponse by {model}"
        
        # Store assistant's response in memory
        self.memory.add_interaction("assistant", response_text, model, service)
        
        return response_text

    # ... (rest of existing methods remain the same)

@click.group()
@click.pass_context
def cli(ctx):
    """AI CLI - Interact with multiple AI models with persistent memory"""
    ctx.obj = AITools()

@cli.command()
@click.option('--service', default=None, help='AI service to use (anthropic/openai)')
@click.option('--model', help='Override default model')
@click.option('--temperature', type=float, help='Override default temperature')
@click.option('--raw', is_flag=True, help='Output raw text without markdown rendering')
@click.option('--computer-control', is_flag=True, help='Use computer control capabilities (Anthropic beta)')
@click.option('--smart', is_flag=True, help='Use GPT-4o model')
@click.option('--complex', is_flag=True, help='Use Claude-3-Opus model')
@click.option('--file', '-f', multiple=True, type=click.Path(exists=True), help='File(s) to use as reference')
@click.option('--folder', '-d', type=click.Path(exists=True, file_okay=False, dir_okay=True), help='Folder containing reference files')
@click.argument('prompt', required=False)
@click.pass_obj
def ask(ai_tools, service, model, temperature, raw, computer_control, smart, complex, file, folder, prompt):
    """Send a prompt to a chat model with persistent memory and file references"""
    service = service or ai_tools.config['default_service']
    
    if smart:
        service = 'openai'
        model = 'gpt-4o'
    elif complex:
        service = 'anthropic'
        model = 'claude-3-opus-20240229'
    else:
        model = model or ai_tools.config[service]['default_model']
    
    # Process file inputs
    file_contents = []
    if file:
        for f in file:
            with open(f, 'r') as file_input:
                file_contents.append(f"File: {f}\n\n{file_input.read()}\n\n")
    
    if folder:
        for root, _, files in os.walk(folder):
            for f in files:
                file_path = os.path.join(root, f)
                with open(file_path, 'r') as file_input:
                    file_contents.append(f"File: {file_path}\n\n{file_input.read()}\n\n")
    
    # Combine file contents with the prompt
    if file_contents:
        file_content_str = "\n".join(file_contents)
        prompt = f"Reference Files:\n\n{file_content_str}\n\nUser Query: {prompt}"
    
    if not prompt:
        prompt = click.edit(text="Enter your prompt here:\n")
        if prompt is None:
            return
    
    try:
        response = ai_tools.get_chat_response(
            prompt, service, model, temperature,
            computer_control=computer_control
        )
        
        if raw:
            click.echo(response)
        else:
            console.print(Markdown(response))
    
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if "--debug" in os.getenv('AI_TOOLS_OPTIONS', ''):
            raise

@cli.command()
@click.argument('query')
@click.pass_obj
def search(ai_tools, query):
    """Search through memory"""
    results = ai_tools.memory.search_memory(query)
    
    if not results:
        console.print("[yellow]No matching memories found[/yellow]")
        return
    
    table = Table(show_header=True)
    table.add_column("Timestamp")
    table.add_column("Role")
    table.add_column("Content")
    table.add_column("Model")
    
    for result in results:
        table.add_row(
            result['timestamp'],
            result['role'],
            result['content'][:100] + "..." if len(result['content']) > 100 else result['content'],
            result['model']
        )
    
    console.print(table)

@cli.command()
@click.option('--limit', default=10, help='Number of recent interactions to show')
@click.pass_obj
def show_history(ai_tools, limit):
    """Show recent interaction history"""
    with sqlite3.connect(ai_tools.config['db_path']) as conn:
        results = conn.execute(
            """
            SELECT timestamp, role, content, model, service 
            FROM memory 
            ORDER BY timestamp DESC 
            LIMIT ?
            """,
            (limit,)
        ).fetchall()
    
    table = Table(show_header=True)
    table.add_column("Timestamp")
    table.add_column("Role")
    table.add_column("Content")
    table.add_column("Model")
    
    for row in reversed(results):  # Show in chronological order
        table.add_row(
            datetime.fromtimestamp(row[0]).strftime('%Y-%m-%d %H:%M:%S'),
            row[1],
            row[2][:100] + "..." if len(row[2]) > 100 else row[2],
            row[3]
        )
    
    console.print(table)

# If image-generate is implemented, add it here
@cli.command()
@click.option('--model', default='dall-e-3', help='Image generation model to use')
@click.option('--size', default='1024x1024', help='Size of the generated image')
@click.argument('prompt')
def image_generate(model, size, prompt):
    """Generate images using AI models

    \b
    Usage:
      ai image-generate [OPTIONS] PROMPT

    Options:
      --model TEXT  Image generation model to use (default: dall-e-3)
      --size TEXT   Size of the generated image (default: 1024x1024)
    """
    click.echo("Image generation not yet implemented.")

if __name__ == '__main__':
    cli()
