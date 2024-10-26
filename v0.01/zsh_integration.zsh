#!/bin/zsh

# AI Tools Configuration
export AI_TOOLS_PATH="${HOME}/.ai-tools"
path=("${AI_TOOLS_PATH}/bin" $path)

# Activate Python virtual environment
source "${AI_TOOLS_PATH}/venv/bin/activate" 2>/dev/null

# AI Tools functions
function ai() {
    "${AI_TOOLS_PATH}/bin/ai-cli.py" "$@"
}

function gpt() {
    "${AI_TOOLS_PATH}/bin/ai-cli.py" ask --service openai "$@"
}

function opus() {
    "${AI_TOOLS_PATH}/bin/ai-cli.py" ask --service anthropic --model claude-3-opus-20240229 "$@"
}

function sonnet() {
    "${AI_TOOLS_PATH}/bin/ai-cli.py" ask --service anthropic --model claude-3-sonnet-20240229 "$@"
}

function haiku() {
    "${AI_TOOLS_PATH}/bin/ai-cli.py" ask --service anthropic --model claude-3-haiku-20240307 "$@"
}

function claude2() {
    "${AI_TOOLS_PATH}/bin/ai-cli.py" ask --service anthropic --model claude-2.1 "$@"
}

# Claude Computer Control shortcuts
function claudec() {
    "${AI_TOOLS_PATH}/bin/ai-cli.py" ask --service anthropic --computer-control --model claude-3-opus-20240229 "$@"
}

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

# Advanced history search with fzf
function ai_history() {
    local selected
    selected=$(cat "${AI_TOOLS_PATH}/cache/history.jsonl" | jq -r '.prompt' | fzf --height 40% --reverse)
    if [[ -n "$selected" ]]; then
        BUFFER="ai \"$selected\""
        CURSOR=$#BUFFER
        zle redisplay
    fi
}
zle -N ai_history
bindkey '^X^H' ai_history

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
    
    echo "\n[Computer Control Models]"
    echo "claudec    : Claude with computer control (Beta)"
    
    echo "\nUsage examples:"
    echo "gpt4o \"What is the meaning of life\"     # Use GPT-4o"
    echo "opus \"Tell me about quantum physics\"    # Use Claude 3 Opus"
    echo "claudec \"Open Chrome\"                   # Use Claude with computer control"
}
