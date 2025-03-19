import streamlit as st
import os
import re
import numpy as np
import nest_asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
nest_asyncio.apply()

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

credentials = {
    "api_key": os.environ["OPENAI_API_KEY"],
}
print(credentials)
# Define the available LLM settings as a dictionary.
available_models = {
    "GPT4oMini": (
        ChatOpenAI(**credentials, model="gpt-4o-mini", temperature=0.0),
        OpenAIEmbeddings(**credentials, model="text-embedding-ada-002", chunk_size=16),
    ),
    "GPT4o": (
        ChatOpenAI(**credentials, model="gpt-4o", temperature=0.0),
        OpenAIEmbeddings(**credentials, model="text-embedding-ada-002", chunk_size=16),
    )
}
print(available_models['GPT4o'][0].invoke("Hello, world!"))
# Use tabs to separate Game and Settings.
tabs = st.tabs(["Game", "Settings"])

# Ensure default values if not already set.
if "llm_model" not in st.session_state:
    st.session_state.llm_model = available_models["GPT4oMini"][0]
if "N_TURNS" not in st.session_state:
    st.session_state.N_TURNS = 10
if "disable_reasoning" not in st.session_state:
    st.session_state.disable_reasoning = False
if "expose_reasoning" not in st.session_state:
    st.session_state.expose_reasoning = False
if "opponent_strategy" not in st.session_state:
    st.session_state.opponent_strategy = "Random"  # Default strategy
if "last_ai_move" not in st.session_state:
    st.session_state.last_ai_move = None

# Inject custom CSS for blank squares with flashing animations.
st.markdown("""
    <style>
    .square {
        display: inline-block;
        width: 80px;
        height: 80px;
        font-size: 48px;
        text-align: center;
        line-height: 80px;
        border: 2px solid #ccc;
        border-radius: 10px;
        margin: 5px;
    }
    .flash-green {
        animation: flashGreen 1s;
        background-color: #90EE90;
    }
    .flash-red {
        animation: flashRed 1s;
        background-color: #FFCCCB;
    }
    @keyframes flashGreen {
        0% { background-color: white; }
        50% { background-color: #90EE90; }
        100% { background-color: white; }
    }
    @keyframes flashRed {
        0% { background-color: white; }
        50% { background-color: #FFCCCB; }
        100% { background-color: white; }
    }
    </style>
    """, unsafe_allow_html=True)

# Prompt templates.
REASONING_MODE = """
You must reason as to what action you should take. Your reasoning should be within the 
<reasoning>
</reasoning>
tags.
"""

ANSWER_FORMAT_EXAMPLE = """
Your answer should be within the 
<answer>
</answer>
tags.
"""

TASK_DEFINITION = """
Your task is as follows:
You are playing a multi-turn game against a single opponent. There are {n_turns} turns.
There is a pot of money (10 points). You can choose to take or share it.
Scoring:
- take vs share: you get 10 points.
- both take: 1 point each.
- both share: 5 points each.
- share vs take: opponent gets 10 points.
Current game state:
<state>
You: {your_points} points | Opponent: {their_points} points
</state>
<turn>
{turn}
</turn>
<history>
{history}
</history>
It is your turn. What will you do?
"""

def extract_reasoning(response_text):
    pattern = r"<reasoning>(.*?)</reasoning>"
    match = re.search(pattern, response_text, re.DOTALL)
    return match.group(1).strip() if match else "No reasoning provided."

def extract_answer(response_text):
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip().lower()
    else:
        raise ValueError("Invalid response: missing <answer> tags.")

# Helper function for opponent move.
def get_opponent_move():
    if st.session_state.opponent_strategy == "Tit-for-tat":
        # Use last_ai_move if available; otherwise, choose randomly.
        if st.session_state.last_ai_move in ["take", "share"]:
            return st.session_state.last_ai_move
        else:
            return np.random.choice(["take", "share"])
    elif st.session_state.opponent_strategy == 'AI':
        _, move = get_ai_move(update_ui=False)
        return move
    elif st.session_state.opponent_strategy == 'Always Take':
        return "take"
    elif st.session_state.opponent_strategy == 'Always Share':
        return "share"
    else:
        return np.random.choice(["take", "share"])


# Initialize session state for game.
if "turn" not in st.session_state:
    st.session_state.turn = 1
if "your_points" not in st.session_state:
    st.session_state.your_points = 0
if "their_points" not in st.session_state:
    st.session_state.their_points = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "ai_move" not in st.session_state:
    st.session_state.ai_move = None
if "opp_move" not in st.session_state:
    st.session_state.opp_move = None

# Placeholders.
game_state_placeholder = st.empty()
decision_placeholder = st.empty()
game_history_placeholder = st.empty()

def render_game_state():
    with game_state_placeholder.container():
        st.subheader(f"Turn {st.session_state.turn} of {st.session_state.N_TURNS}")
        st.write(f"**Your points:** {st.session_state.your_points}")
        st.write(f"**Opponent points:** {st.session_state.their_points}")
        ai, cash, opp = st.columns(3)
        ai.markdown(render_emoji_square(st.session_state.ai_move, "ai"), unsafe_allow_html=True)
        cash.markdown("<div style='text-align: center; font-size: 64px;'>💰</div>", unsafe_allow_html=True)
        opp.markdown(render_emoji_square(st.session_state.opp_move, "opponent"), unsafe_allow_html=True)
    with game_history_placeholder.container():
        with st.expander("Game History", expanded=False):
            for line in st.session_state.history:
                st.write(line)

def render_emoji_square(decision, player):
    decision = decision if decision in ["take", "share"] else "take"
    css_class = "square flash-green" if decision == "share" else "square flash-red"
    emoji = "🤖" if player == "ai" else "👤"
    return f"<div class='{css_class}'>{emoji}</div>"

def render_decision(ai_reasoning, ai_move, opp_move):
    with decision_placeholder.container():
        col_ai, col_opp = st.columns(2)
        with col_ai:
            with st.expander("AI Decision", expanded=False):
                st.write("AI Move:")
                st.write(ai_move)
                st.markdown(render_emoji_square(ai_move, "ai"), unsafe_allow_html=True)
                if not st.session_state.disable_reasoning:
                    st.write("AI Reasoning:")
                    st.write(ai_reasoning)
        with col_opp:
            with st.expander("Opponent Decision", expanded=False):
                st.write("Opponent Move:")
                st.write(opp_move)
                st.markdown(render_emoji_square(opp_move, "opponent"), unsafe_allow_html=True)

def get_ai_move(update_ui=True):
    # --- Step 1: Build prompt (without current opponent move) ---
    reasoning_prompt = "" if st.session_state.disable_reasoning else REASONING_MODE
    exposure_message = ""
    if st.session_state.expose_reasoning:
        exposure_message = "The opponent will be able to see your reasoning, but not your final answer, so be aware that your thought process is being watched! Don't forget that everything you are saying will be visible to the opponent and may affect how they choose to either take or share!"
    prompt = TASK_DEFINITION.format(
        n_turns=st.session_state.N_TURNS,
        your_points=st.session_state.your_points,
        their_points=st.session_state.their_points,
        history="\n".join(st.session_state.history),
        turn=st.session_state.turn
    ) + "\n" + reasoning_prompt + "\n" + exposure_message + "\n" + ANSWER_FORMAT_EXAMPLE

    response = st.session_state.llm_model.invoke(prompt)
    if hasattr(response, "content"):
        response_text = response.content
    else:
        if update_ui:
            st.error("Unexpected LLM response format.")
        return None

    try:
        ai_reasoning = extract_reasoning(response_text) if not st.session_state.disable_reasoning else ""
        ai_move = extract_answer(response_text)
    except ValueError as e:
        if update_ui:
            st.error(f"Error parsing response: {e}")
        ai_reasoning = "Error: could not extract reasoning."
        ai_move = "take"
    return ai_reasoning, ai_move

def play_turn(update_ui=True):
    ai_reasoning, ai_move = get_ai_move(update_ui=update_ui)
    # --- Store the previous AI move before updating ---
    if st.session_state.ai_move is not None:
        st.session_state.last_ai_move = st.session_state.ai_move
    st.session_state.ai_move = ai_move

    # --- Step 2: Now generate opponent move based on chosen strategy ---
    opp_move = get_opponent_move()
    st.session_state.opp_move = opp_move

    # --- Step 3: Update scores based on both moves ---
    if ai_move == "take" and opp_move == "take":
        st.session_state.your_points += 1
        st.session_state.their_points += 1
    elif ai_move == "take" and opp_move == "share":
        st.session_state.your_points += 10
    elif ai_move == "share" and opp_move == "take":
        st.session_state.their_points += 10
    elif ai_move == "share" and opp_move == "share":
        st.session_state.your_points += 5
        st.session_state.their_points += 5
    else:
        st.warning("Unexpected move combination.")

    # --- Step 4: Update history and increment turn ---
    st.session_state.history.append(f"You: {ai_move}")
    st.session_state.history.append(f"Opponent: {opp_move}")
    st.session_state.turn += 1

    # --- Step 5: Re-render game state and decisions ---
    if update_ui:
        render_game_state()
        render_decision(ai_reasoning, ai_move, opp_move)
    return ai_reasoning

def skip_to_end():
    with st.spinner("Skipping to end..."):
        while st.session_state.turn < st.session_state.N_TURNS:
            play_turn(update_ui=True)
        final_reasoning = play_turn(update_ui=True)
    return final_reasoning

def game_tab():
    global game_state_placeholder, decision_placeholder, game_history_placeholder
    game_state_placeholder = st.empty()
    decision_placeholder = st.empty()
    game_history_placeholder = st.empty()

    render_game_state()
    
    # Game control buttons.
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("Play Next Turn"):
            if st.session_state.turn > st.session_state.N_TURNS:
                st.info("Game over! Reset to start again.")
            else:
                play_turn(update_ui=True)
    with col_btn2:
        if st.button("Skip to End"):
            if st.session_state.turn > st.session_state.N_TURNS:
                st.info("Game already over.")
            else:
                skip_to_end()
    with col_btn3:
        if st.button("Reset Game"):
            st.session_state.turn = 0
            st.session_state.your_points = 0
            st.session_state.their_points = 0
            st.session_state.history = []
            st.rerun()

    if st.session_state.turn > st.session_state.N_TURNS:
        st.subheader("Final Scores")
        st.write(f"**Your points:** {st.session_state.your_points}")
        st.write(f"**Opponent points:** {st.session_state.their_points}")

def settings_tab():
    st.header("Settings")
    model_choice = st.selectbox(
        "Select the LLM model to use:",
        options=list(available_models.keys()),
        index=0,
    )
    st.write("Selected model:", model_choice)
    st.session_state.llm_model = available_models[model_choice][0]
    num_turns = st.number_input("Set number of turns:", min_value=1, value=st.session_state.N_TURNS, step=1)
    st.session_state.N_TURNS = num_turns
    disable_reasoning = st.checkbox("Prevent AI from providing reasoning", value=st.session_state.disable_reasoning)
    st.session_state.disable_reasoning = disable_reasoning
    expose_reasoning = st.checkbox("Tell AI that opponent sees your reasoning", value=st.session_state.expose_reasoning)
    st.session_state.expose_reasoning = expose_reasoning
    opp_strategy = st.selectbox("Select opponent strategy:", options=["Random", "Tit-for-tat", "AI", "Always Take", "Always Share"], index=0)
    st.session_state.opponent_strategy = opp_strategy

with tabs[0]:
    game_tab()
with tabs[1]:
    settings_tab()
