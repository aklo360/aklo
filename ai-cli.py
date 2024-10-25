#!/usr/bin/env python3

# Existing imports
import os
os.environ['PYTHONWARNINGS'] = 'ignore:Unverified HTTPS request'
import warnings
warnings.filterwarnings('ignore', message='.*OpenSSL.*')

import click
import yaml
import json
from datetime import datetime
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
                'default_model': os.getenv('AI_OPENAI_DEFAULT_MODEL', 'gpt-4'),
                'max_tokens': int(os.getenv('AI_MAX_TOKENS', '4096'))
            },
            'anthropic': {
                'default_model': os.getenv('AI_ANTHROPIC_DEFAULT_MODEL', 'claude-3-opus-20240229'),
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
        
        # Add current prompt to context
        context.append({"role": "user", "content": prompt})
        
        # Store user message in memory
        self.memory.add_interaction("user", prompt, model, service)
        
        if service == 'anthropic':
            client = Anthropic()
            
            if computer_control:
                response_text = "Computer Control API is not yet available."
            else:
                response = client.messages.create(
                    model=model,
                    max_tokens=self.config['anthropic']['max_tokens'],
                    temperature=temp,
                    messages=context
                )
                response_text = str(response.content[0].text) if hasattr(response.content, '__getitem__') else str(response.content)
        
        elif service == 'openai':
            response = self.client.chat.completions.create(
                model=model,
                messages=context,
                temperature=temp
            )
            response_text = response.choices[0].message.content
        
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
@click.argument('prompt', required=False)
@click.pass_obj
def ask(ai_tools: AITools, service: Optional[str], model: Optional[str],
        temperature: Optional[float], raw: bool, computer_control: bool,
        prompt: Optional[str]):
    """Send a prompt to a chat model with persistent memory"""
    service = service or ai_tools.config['default_service']
    used_model = model or ai_tools.config[service]['default_model']
    
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
def search(ai_tools: AITools, query: str):
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
def show_history(ai_tools: AITools, limit: int):
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

if __name__ == '__main__':
    cli()
