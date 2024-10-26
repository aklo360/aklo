#!/usr/bin/env python3

# Import order matters for SSL warning suppression
import os
os.environ['PYTHONWARNINGS'] = 'ignore:Unverified HTTPS request'
import warnings
warnings.filterwarnings('ignore', message='.*OpenSSL.*')

import click
import yaml
import json
import base64
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
import requests
import urllib3
import sqlite3
import time
from rich.progress import Progress, SpinnerColumn, TextColumn

urllib3.disable_warnings()

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
        messages = []
        total_chars = 0
        
        with sqlite3.connect(self.db_path) as conn:
            for row in conn.execute(
                "SELECT role, content FROM memory ORDER BY timestamp DESC LIMIT ?",
                (self.context_window,)
            ):
                message = {"role": row[0], "content": row[1]}
                message_chars = len(row[1])
                
                if total_chars + message_chars > max_chars:
                    break
                
                messages.insert(0, message)
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
                'default_model': os.getenv('AI_OPENAI_DEFAULT_MODEL', 'gpt-4o'),
                'max_tokens': int(os.getenv('AI_MAX_TOKENS', '4096'))
            },
            'anthropic': {
                'default_model': os.getenv('AI_ANTHROPIC_DEFAULT_MODEL', 'claude-3-opus-20240229'),
                'max_tokens': int(os.getenv('AI_MAX_TOKENS', '4096'))
            },
            'default_service': os.getenv('AI_DEFAULT_SERVICE', 'anthropic'),
            'temperature': float(os.getenv('AI_DEFAULT_TEMPERATURE', '0.7')),
            'db_path': os.path.expanduser('~/.ai-tools/memory.db'),
            'context_window': int(os.getenv('AI_CONTEXT_WINDOW', '20'))
        }
        
        # Ensure database directory exists
        Path(self.config['db_path']).parent.mkdir(parents=True, exist_ok=True)
        
        self.memory = MemoryManager(self.config['db_path'], self.config['context_window'])
        self.client = openai.OpenAI()

    def save_to_history(self, prompt: str, response: str, service: str, model: str, type: str = "chat"):
        # Save to memory database
        self.memory.add_interaction("user", prompt[:1000], model, service)
        self.memory.add_interaction("assistant", response[:1000], model, service)

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
                    response_text = "Computer Control API is not yet available or not properly configured."
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
        
        return response_text

    def generate_image(self, prompt: str, model: str = "dall-e-3", size: str = "1024x1024", 
                      quality: str = "standard") -> str:
        try:
            response = self.client.images.generate(
                model=model,
                prompt=prompt,
                n=1,
                size=size,
                quality=quality
            )
            
            image_url = response.data[0].url
            self.save_to_history(prompt, image_url, "openai", model, "image")
            return image_url
        except Exception as e:
            console.print(f"[red]Error generating image:[/red] {str(e)}")
            console.print("\n[yellow]Debug information:[/yellow]")
            console.print(f"Prompt: {prompt}")
            console.print(f"Model: {model}")
            console.print(f"Size: {size}")
            console.print(f"Quality: {quality}")
            if "--debug" in os.getenv('AI_TOOLS_OPTIONS', ''):
                raise
            return None

    def text_to_speech(self, text: str, model: str = "tts-1", voice: str = "alloy", 
                      output_file: str = "output.mp3"):
        try:
            response = self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text
            )
            
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            self.save_to_history(text, output_file, "openai", model, "speech")
            return output_file
        except Exception as e:
            console.print(f"[red]Error converting text to speech:[/red] {str(e)}")
            console.print("\n[yellow]Debug information:[/yellow]")
            console.print(f"Text: {text}")
            console.print(f"Model: {model}")
            console.print(f"Voice: {voice}")
            console.print(f"Output file: {output_file}")
            if "--debug" in os.getenv('AI_TOOLS_OPTIONS', ''):
                raise
            return None

    def transcribe_audio(self, file_path: str, model: str = "whisper-1", 
                        language: Optional[str] = None) -> str:
        try:
            with open(file_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    language=language
                )
            
            self.save_to_history(file_path, response.text, "openai", model, "transcription")
            return response.text
        except Exception as e:
            console.print(f"[red]Error transcribing audio:[/red] {str(e)}")
            console.print("\n[yellow]Debug information:[/yellow]")
            console.print(f"File path: {file_path}")
            console.print(f"File exists: {os.path.exists(file_path)}")
            console.print(f"File size: {os.path.getsize(file_path)} bytes")
            console.print(f"Model: {model}")
            console.print(f"Language: {language}")
            if "--debug" in os.getenv('AI_TOOLS_OPTIONS', ''):
                raise
            return None

    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> list:
        response = self.client.embeddings.create(
            model=model,
            input=text
        )
        
        embedding = response.data[0].embedding
        self.save_to_history(text, str(len(embedding)) + " dimensions", "openai", model, "embedding")
        return embedding

    def moderate_content(self, text: str, model: str = "text-moderation-latest") -> dict:
        try:
            response = self.client.moderations.create(
                model=model,
                input=text
            )
            
            results = response.results[0]
            
            # Convert the Pydantic model to a dictionary using model_dump()
            results_dict = results.model_dump()
            
            return {
                "flagged": results_dict['flagged'],
                "categories": results_dict['categories'],
                "scores": results_dict['category_scores']
            }
                
        except Exception as e:
            console.print("[yellow]Debug info:[/yellow]")
            try:
                # Print more detailed debug info
                console.print(f"Results type: {type(results)}")
                console.print(f"Available methods: {dir(results)}")
                results_dict = results.model_dump()
                console.print(f"Full results structure: {results_dict}")
            except:
                pass
            raise Exception(f"Moderation error: {str(e)}")

@click.group()
@click.pass_context
def cli(ctx):
    """AI CLI - Interact with multiple AI models from your terminal"""
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
def ask(ai_tools: AITools, service: Optional[str], model: Optional[str], 
        temperature: Optional[float], raw: bool, computer_control: bool, smart: bool, complex: bool,
        file: Tuple[str], folder: Optional[str], prompt: Optional[str]):
    """Send a prompt to a chat model"""
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
            prompt, service, model, temperature, computer_control=computer_control
        )
        ai_tools.save_to_history(prompt, response, service, model)
        
        if raw:
            click.echo(response)
        else:
            console.print(Markdown(response))
    
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if "--debug" in os.getenv('AI_TOOLS_OPTIONS', ''):
            raise

@cli.command()
@click.option('--model', default="dall-e-3", help='Image generation model to use')
@click.option('--size', default="1024x1024", help='Image size (e.g., 256x256, 512x512, 1024x1024)')
@click.option('--quality', default="standard", help='Image quality (standard or enhanced)')
@click.argument('prompt')
@click.pass_obj
def image_gen(ai_tools: AITools, model: str, size: str, quality: str, prompt: str):
    """Generate an image from a text prompt"""
    try:
        image_url = ai_tools.generate_image(prompt, model, size, quality)
        console.print(f"[green]Generated image:[/green] {image_url}")
    except Exception as e:
        console.print(f"[red]Error generating image:[/red] {str(e)}")

@cli.command()
@click.option('--model', default="tts-1", help='Text-to-speech model to use')
@click.option('--voice', default="alloy", help='Voice to use for speech')
@click.option('--output', default="output.mp3", help='Output file name')
@click.argument('text')
@click.pass_obj
def speech(ai_tools: AITools, model: str, voice: str, output: str, text: str):
    """Convert text to speech"""
    try:
        output_file = ai_tools.text_to_speech(text, model, voice, output)
        console.print(f"[green]Speech saved to:[/green] {output_file}")
    except Exception as e:
        console.print(f"[red]Error converting text to speech:[/red] {str(e)}")

@cli.command()
@click.option('--model', default="whisper-1", help='Whisper model to use')
@click.option('--language', help='Specify language code (optional)')
@click.argument('file_path', type=click.Path(exists=True))
@click.pass_obj
def transcribe(ai_tools: AITools, model: str, language: Optional[str], file_path: str):
    """Transcribe audio to text"""
    try:
        text = ai_tools.transcribe_audio(file_path, model, language)
        console.print(Markdown(text))
    except Exception as e:
        console.print(f"[red]Error transcribing audio:[/red] {str(e)}")

@cli.command()
@click.option('--model', default="text-embedding-3-large", help='Embedding model to use')
@click.argument('text')
@click.pass_obj
def embed(ai_tools: AITools, model: str, text: str):
    """Generate text embeddings"""
    try:
        embedding = ai_tools.get_embedding(text, model)
        console.print(f"[green]Generated embedding with {len(embedding)} dimensions[/green]")
        if click.confirm("Do you want to see the raw embedding values?"):
            console.print(embedding)
    except Exception as e:
        console.print(f"[red]Error generating embedding:[/red] {str(e)}")

@cli.command()
@click.option('--model', default="text-moderation-latest", help='Moderation model to use')
@click.argument('text')
@click.pass_obj
def moderate(ai_tools: AITools, model: str, text: str):
    """Moderate content"""
    try:
        results = ai_tools.moderate_content(text, model)
        
        # Create a styled table
        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            title="Content Moderation Results"
        )
        
        # Add columns with proper styling
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Flagged", style="red", justify="center")
        table.add_column("Score", style="green", justify="right")
        
        # Add rows
        categories = results["categories"]
        scores = results["scores"]
        
        for category in categories:
            score_value = scores.get(category, 0.0)  # Default to 0.0 if score is None
            if score_value is None:
                score_value = 0.0
                
            table.add_row(
                category.replace('/', ' / '),  # Make category names more readable
                "🚫" if categories[category] else "✓",
                f"{float(score_value):.4f}"
            )
        
        # Print results with styling
        console.print()
        console.print("[bold]Overall Status:[/bold]", 
                     "[red]⚠️  Flagged[/red]" if results["flagged"] else "[green]✓ Safe[/green]")
        console.print()
        console.print(table)
        console.print()
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        # Add more detailed error information
        console.print("\n[yellow]Debug information:[/yellow]")
        console.print(f"Results structure: {results if 'results' in locals() else 'Not available'}")
        if "--debug" in os.getenv('AI_TOOLS_OPTIONS', ''):
            raise

@cli.command()
@click.option('--service', default=None, help='AI service to list models for (anthropic/openai)')
@click.pass_obj
def list_models(ai_tools: AITools, service: Optional[str]):
    """List available models"""
    service = service or ai_tools.config['default_service']
    try:
        models = ai_tools.list_models(service)
        console.print(f"[green]Available models for {service}:[/green]")
        for model in models:
            console.print(f"- {model}")
    except Exception as e:
        console.print(f"[red]Error listing models:[/red] {str(e)}")

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

