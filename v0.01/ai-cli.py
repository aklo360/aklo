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
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from anthropic import Anthropic
import openai
from typing import Optional, Dict, Any
import requests
import urllib3
urllib3.disable_warnings()

console = Console()

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
            'history_file': os.path.expanduser('~/.ai-tools/cache/history.jsonl')
        }
        self.setup_history_file()
        self.client = openai.OpenAI()

    def setup_history_file(self):
        history_path = Path(self.config['history_file'])
        history_path.parent.mkdir(parents=True, exist_ok=True)
        if not history_path.exists():
            history_path.touch()

    def save_to_history(self, prompt: str, response: str, service: str, model: str, type: str = "chat"):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'response': str(response),
            'service': service,
            'model': model,
            'type': type
        }
        
        with open(self.config['history_file'], 'a') as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    def get_chat_response(self, prompt: str, service: str, model: Optional[str] = None, 
                         temperature: Optional[float] = None, computer_control: bool = False) -> str:
        if service == 'anthropic':
            client = Anthropic()
            model = model or self.config['anthropic']['default_model']
            temp = temperature or self.config['temperature']
            
            if computer_control:
                try:
                    # Placeholder for computer control API
                    return "Computer Control API is not yet available or not properly configured."
                except ImportError:
                    return "Computer Control API is not yet available."
            else:
                response = client.messages.create(
                    model=model,
                    max_tokens=self.config['anthropic']['max_tokens'],
                    temperature=temp,
                    messages=[{"role": "user", "content": prompt}]
                )
                return str(response.content[0].text) if hasattr(response.content, '__getitem__') else str(response.content)
        
        elif service == 'openai':
            model = model or self.config['openai']['default_model']
            temp = temperature or self.config['temperature']
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp
            )
            return response.choices[0].message.content

    def generate_image(self, prompt: str, model: str = "dall-e-3", size: str = "1024x1024", 
                      quality: str = "standard") -> str:
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

    def text_to_speech(self, text: str, model: str = "tts-1", voice: str = "alloy", 
                      output_file: str = "output.mp3"):
        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text
        )
        
        response.stream_to_file(output_file)
        self.save_to_history(text, output_file, "openai", model, "speech")
        return output_file

    def transcribe_audio(self, file_path: str, model: str = "whisper-1", 
                        language: Optional[str] = None) -> str:
        with open(file_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                language=language
            )
        
        self.save_to_history(file_path, response.text, "openai", model, "transcription")
        return response.text

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
@click.argument('prompt', required=False)
@click.pass_obj
def ask(ai_tools: AITools, service: Optional[str], model: Optional[str], 
        temperature: Optional[float], raw: bool, computer_control: bool, prompt: Optional[str]):
    """Send a prompt to a chat model"""
    service = service or ai_tools.config['default_service']
    used_model = model or ai_tools.config[service]['default_model']
    
    if not prompt:
        prompt = click.edit(text="Enter your prompt here:\n")
        if prompt is None:
            return
    
    try:
        response = ai_tools.get_chat_response(
            prompt, service, model, temperature, computer_control=computer_control
        )
        ai_tools.save_to_history(prompt, response, service, used_model)
        
        if raw:
            click.echo(response)
        else:
            console.print(Markdown(response))
    
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if "--debug" in os.getenv('AI_TOOLS_OPTIONS', ''):
            raise

@cli.command()
@click.option('--model', default="dall-e-3", help='DALL-E model to use')
@click.option('--size', default="1024x1024", help='Image size')
@click.option('--quality', default="standard", help='Image quality')
@click.argument('prompt')
@click.pass_obj
def image_gen(ai_tools: AITools, model: str, size: str, quality: str, prompt: str):
    """Generate an image using DALL-E"""
    try:
        image_url = ai_tools.generate_image(prompt, model, size, quality)
        console.print(f"[green]Image generated:[/green] {image_url}")
    except Exception as e:
        console.print(f"[red]Error generating image:[/red] {str(e)}")

@cli.command()
@click.option('--model', default="tts-1", help='TTS model to use')
@click.option('--voice', default="alloy", help='Voice to use')
@click.option('--output', default="output.mp3", help='Output file path')
@click.argument('text')
@click.pass_obj
def speech(ai_tools: AITools, model: str, voice: str, output: str, text: str):
    """Convert text to speech"""
    try:
        output_file = ai_tools.text_to_speech(text, model, voice, output)
        console.print(f"[green]Audio saved to:[/green] {output_file}")
    except Exception as e:
        console.print(f"[red]Error generating speech:[/red] {str(e)}")

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
@click.pass_obj
def list_models(ai_tools: AITools):
    """List all available AI models"""
    try:
        models = ai_tools.client.models.list()
        
        table = Table(show_header=True)
        table.add_column("Model ID")
        table.add_column("Type")
        table.add_column("Created")
        
        for model in models:
            table.add_row(
                model.id,
                model.object,
                str(datetime.fromtimestamp(model.created))
            )
        
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing models:[/red] {str(e)}")

if __name__ == '__main__':
    cli()
