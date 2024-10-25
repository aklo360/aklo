#!/bin/zsh

# AI Tools Configuration
export AI_TOOLS_PATH="${HOME}/.ai-tools"
path=("${AI_TOOLS_PATH}/bin" $path)

# Set default context window size (adjust as needed)
export AI_CONTEXT_WINDOW="50"

# AI Tools functions - using global python3
function ai() {
    python3 "${AI_TOOLS_PATH}/bin/ai-cli.py" "$@"
}

# GPT-4o Series
function gpt4o() {
    python3 "${AI_TOOLS_PATH}/bin/ai-cli.py" ask --service openai --model gpt-4o "$@"
}

function gpt4omini() {
    python3 "${AI_TOOLS_PATH}/bin/ai-cli.py" ask --service openai --model gpt-4o-mini "$@"
}

function gpt4olatest() {
    python3 "${AI_TOOLS_PATH}/bin/ai-cli.py" ask --service openai --model chatgpt-4o-latest "$@"
}

# Regular GPT commands
function gpt() {
    python3 "${AI_TOOLS_PATH}/bin/ai-cli.py" ask --service openai "$@"
}

function opus() {
    python3 "${AI_TOOLS_PATH}/bin/ai-cli.py" ask --service anthropic --model claude-3-opus-20240229 "$@"
}

function sonnet() {
    python3 "${AI_TOOLS_PATH}/bin/ai-cli.py" ask --service anthropic --model claude-3-sonnet-20240229 "$@"
}

function haiku() {
    python3 "${AI_TOOLS_PATH}/bin/ai-cli.py" ask --service anthropic --model claude-3-haiku-20240307 "$@"
}

function claude2() {
    python3 "${AI_TOOLS_PATH}/bin/ai-cli.py" ask --service anthropic --model claude-2.1 "$@"
}

# Claude Computer Control shortcuts
function claudec() {
    python3 "${AI_TOOLS_PATH}/bin/ai-cli.py" ask --service anthropic --computer-control --model claude-3-opus-20240229 "$@"
}

# New memory-related functions
function ai_history() {
    python3 "${AI_TOOLS_PATH}/bin/ai-cli.py" show-history "$@"
}

function ai_search() {
    python3 "${AI_TOOLS_PATH}/bin/ai-cli.py" search "$@"
}

# Shorthand for common memory operations
alias aih='ai_history'
alias ais='ai_search'

# Advanced history search with fzf
function ai_search_fzf() {
    local selected
    selected=$(ai_history --limit 1000 | fzf --height 40% --reverse --ansi)
    if [[ -n "$selected" ]]; then
        BUFFER="ai \"${selected##*|}\""  # Extract the content part
        CURSOR=$#BUFFER
        zle redisplay
    fi
}
zle -N ai_search_fzf
bindkey '^X^F' ai_search_fzf

# Key bindings for AI tools
bindkey '^X^A' _ai_widget

# Custom widget for AI prompt
function _ai_widget() {
    local cmd="ai"
    BUFFER="${BUFFER:0:$CURSOR}$cmd ${BUFFER:$CURSOR}"
    CURSOR=$(( CURSOR + ${#cmd} + 1 ))
    zle redisplay
}
zle -N _ai_widget

# Completions
fpath=("${AI_TOOLS_PATH}/completions" $fpath)

# Model information function
function ai_models() {
    echo "\n[OpenAI GPT-4o Series]"
    echo "gpt4o      : GPT-4o - Latest flagship model (gpt-4o)"
    echo "gpt4omini  : GPT-4o Mini - Affordable, fast model"
    echo "gpt4olatest: ChatGPT GPT-4o - Latest research model"
    
    echo "\n[OpenAI O1 Series]"
    echo "o1         : o1-preview - Complex reasoning model"
    echo "o1mini     : o1-mini - Faster reasoning model"
    
    echo "\n[OpenAI GPT-4 Series]"
    echo "gpt4t      : GPT-4 Turbo (includes vision capabilities)"
    echo "gpt4       : GPT-4 (gpt-4-0125-preview)"
    echo "gpt4v      : GPT-4 Vision (using Turbo model)"
    
    echo "\n[OpenAI GPT-3.5 Series]"
    echo "gpt35      : GPT-3.5 Turbo"
    echo "gpt35i     : GPT-3.5 Turbo Instruct"
    
    echo "\n[OpenAI Base Models]"
    echo "babbage    : Babbage-002 base model"
    echo "davinci    : Davinci-002 base model"
    
    echo "\n[Anthropic Models]"
    echo "opus       : Claude 3 Opus (Most capable)"
    echo "sonnet     : Claude 3 Sonnet (Balanced)"
    echo "haiku      : Claude 3 Haiku (Fastest)"
    echo "claude2    : Claude 2.1 (Legacy)"
    
    echo "\n[Memory Commands]"
    echo "ai history : Show recent conversation history"
    echo "ai search  : Search through past conversations"
    echo "aih        : Shorthand for ai history"
    echo "ais        : Shorthand for ai search"
    
    echo "\nUsage examples:"
    echo "gpt4o \"What is the meaning of life\"     # Use GPT-4o"
    echo "opus \"Tell me about quantum physics\"    # Use Claude 3 Opus"
    echo "aih --limit 5                           # Show last 5 interactions"
    echo "ais \"python\"                           # Search for mentions of python"
}