import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
import nest_asyncio
import random

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain import LLMChain, PromptTemplate

nest_asyncio.apply()

# Load environment variables for LLM configuration
load_dotenv(find_dotenv())
OPENAI_BASE_URL = "https://genai-sharedservice-emea.pwcinternal.com"
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
os.environ["AZURE_OPENAI_ENDPOINT"] = os.environ["OPENAI_BASE_URL"]
os.environ["AZURE_OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]
os.environ["OPENAI_API_VERSION"] = "2023-06-01-preview"

##### Global Variables #####
credentials = {"api_key": os.environ["AZURE_OPENAI_API_KEY"]}
available_models = {
    "GPT4oMini": (
        AzureChatOpenAI(**credentials, model="azure.gpt-4o-mini", temperature=0.0),
        AzureOpenAIEmbeddings(**credentials, model="azure.text-embedding-ada-002", chunk_size=16),
    ),
    "GPT4o": (
        AzureChatOpenAI(**credentials, model="azure.gpt-4o", temperature=0.0),
        AzureOpenAIEmbeddings(**credentials, model="azure.text-embedding-ada-002", chunk_size=16),
    ),
}

def build_llm(model_name, temperature):
    if model_name == "GPT4oMini":
        return AzureChatOpenAI(**credentials, model="azure.gpt-4o-mini", temperature=temperature)
    else:
        return AzureChatOpenAI(**credentials, model="azure.gpt-4o", temperature=temperature)

# ----- Custom CSS for Chat Messages -----
st.markdown(
    """
    <style>
    div[style="font-size: 2em; line-height: 1.0;"] {
        font-size: 3em !important;
        line-height: 1.8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    [data-testid="stChatMessageContent"] {
        left-margin: 100px !important;
        right-margin: 100px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    [data-testid="stChatMessageContent"][aria-label="Chat message from assistant"] {
        padding: 25px;
        border-radius: 15px;
        background-color: #cfe9ff;
        color: black;
        margin-bottom: 10px;
        max-width: 60%;
        word-wrap: break-word;
    }
    [data-testid="stChatMessageContent"][aria-label="Chat message from user"] {
        padding: 25px;
        border-radius: 15px;
        background-color: #e0e0e0;
        color: black;
        margin-bottom: 10px;
        max-width: 60%;
        word-wrap: break-word;
        align-self: flex-end;
    }
    .st-chat-message-container {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 10px;
    }
    .user-message .st-chat-message-container {
        justify-content: flex-end;
    }
    .assistant-message .st-chat-message-container {
        justify-content: flex-start;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    div[data-testid="chatAvatarIcon-assistant"] {
        flex-direction: row-reverse;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
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
    .agent1-text { color: blue; }
    .agent2-text { color: green; }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# Negotiation Agent Implementation
# ------------------------------
class NegotiationAgent:
    def __init__(self, name: str, negotiation_chain: LLMChain, decision_chain: LLMChain):
        self.name = name
        self.negotiation_chain = negotiation_chain
        self.decision_chain = decision_chain

    def pre_commit_decision(self, context: str) -> str:
        prompt = PromptTemplate(
            input_variables=["agent_name", "context"],
            template=(
                "You are {agent_name}. Based on the current game state and reward settings: {context}, "
                "choose your preliminary move: 'share' or 'take'. Provide only the move as the answer."
            )
        )
        full_prompt = prompt.format(agent_name=self.name, context=context)
        pre_move = self.decision_chain.run(full_prompt)
        return pre_move.strip().lower()

    def negotiate(self, context: str) -> str:
        prompt = PromptTemplate(
            input_variables=["agent_name", "negotiation_history", "context"],
            template=(
                "You are {agent_name}, negotiating a move with your opponent. "
                "Remember: you are allowed to lie, and your goal is to maximize your points. "
                "Here is the conversation so far in the current round: {negotiation_history}\n"
                "Context (including full conversation history, game history, and reward settings): {context}\n"
                "Propose your next negotiation message. If you decide to finalize your move, respond with only 'take' or 'share'."
            )
        )
        full_prompt = prompt.format(agent_name=self.name, negotiation_history="", context=context)
        message = self.negotiation_chain.run(full_prompt)
        return message

    def decide_final_move(self, context: str) -> str:
        prompt = PromptTemplate(
            input_variables=["agent_name", "negotiation_history", "context"],
            template=(
                "You are {agent_name}. After negotiating with your opponent, remember: you are allowed to lie, "
                "and your goal is to maximize your points. "
                "Here is the conversation so far in the current round: {negotiation_history}\n"
                "Context (including full conversation history, game history, and reward settings): {context}\n"
                "Based on the conversation and your internal reasoning, choose your final move: 'share' or 'take'. "
                "Provide only the move as the answer."
            )
        )
        full_prompt = prompt.format(agent_name=self.name, negotiation_history="", context=context)
        final_move = self.decision_chain.run(full_prompt)
        return final_move.strip().lower()

    def private_reasoning(self, context: str) -> str:
        prompt = PromptTemplate(
            input_variables=["agent_name", "context"],
            template=(
                "You are {agent_name}. Think privately about your upcoming negotiation move. "
                "Provide a brief reasoning for your next action without revealing your final decision."
            )
        )
        full_prompt = prompt.format(agent_name=self.name, context=context)
        reasoning = self.decision_chain.run(full_prompt)
        return reasoning.strip()

# ------------------------------
# Helper Function for Graphics
# ------------------------------
def get_emoji(agent):
    return "🤖" if agent == "agent1" else "👤"

def render_emoji_square(decision, agent):
    if decision not in ["take", "share"]:
        decision = "take"
    emoji = get_emoji(agent)
    css_class = "square flash-green" if decision == "share" else "square flash-red"
    return f"<div class='{css_class}'>{emoji}</div>"

# ------------------------------
# LangGraph Implementation for a Negotiation Round (Unified Log)
# ------------------------------
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# The round state now contains a unified negotiation_log list.
class RoundState(TypedDict):
    context: str
    negotiation_log: list  # List of dicts: {"agent": str, "message": str, "reasoning": Optional[str]}
    agent1_pre: str
    agent2_pre: str
    agent1_final: str
    agent2_final: str
    negotiation_step: int
    max_steps: int
    current_turn: str
    agent1_reasoning: str
    agent2_reasoning: str

def choose_initial_turn(state: RoundState):
    turn = random.choice(["agent1", "agent2"])
    st.write(f"[Graph] Initial randomly chosen turn: {turn}")
    return {"current_turn": turn}

def dispatch(state: RoundState):
    return state

def choose_agent_node(state: RoundState, private_reasoning: bool):
    if private_reasoning:
        return "agent1_reasoning_node" if state["current_turn"] == "agent1" else "agent2_reasoning_node"
    else:
        return "agent1_negotiate" if state["current_turn"] == "agent1" else "agent2_negotiate"

def agent1_pre_decision(state: RoundState):
    pre = st.session_state.agent1.pre_commit_decision(state["context"])
    return {"agent1_pre": pre}

def agent2_pre_decision(state: RoundState):
    pre = st.session_state.agent2.pre_commit_decision(state["context"])
    return {"agent2_pre": pre}

def agent1_reasoning_node(state: RoundState):
    reasoning = st.session_state.agent1.private_reasoning(state["context"])
    st.write(f"[Graph] Agent1 private reasoning at step {state['negotiation_step']}: {reasoning}")
    return {"agent1_reasoning": reasoning}

def agent2_reasoning_node(state: RoundState):
    reasoning = st.session_state.agent2.private_reasoning(state["context"])
    st.write(f"[Graph] Agent2 private reasoning at step {state['negotiation_step']}: {reasoning}")
    return {"agent2_reasoning": reasoning}

def agent1_negotiate(state: RoundState):
    if state.get("agent1_final"):
        return {}
    current_round_history = ""
    for entry in state.get("negotiation_log", []):
        current_round_history += f"{entry['agent']}: {entry['message']}\n"
    new_context = state["context"] + "\nCurrent Round History:\n" + current_round_history
    msg = st.session_state.agent1.negotiate(new_context)
    st.write(f"[Graph] Agent1 negotiation step {state['negotiation_step']}: {msg}")
    if msg.strip().lower() in ["take", "share"]:
        return {"agent1_final": msg.strip().lower()}
    else:
        log = state.get("negotiation_log", [])
        reasoning = state.get("agent1_reasoning", "")
        log.append({"agent": "Agent 1", "message": msg, "reasoning": reasoning})
        updated = {
            "negotiation_log": log,
            "negotiation_step": state["negotiation_step"] + 1,
            "current_turn": "agent2",
            "agent1_reasoning": ""
        }
        return updated

def agent2_negotiate(state: RoundState):
    if state.get("agent2_final"):
        return {}
    current_round_history = ""
    for entry in state.get("negotiation_log", []):
        current_round_history += f"{entry['agent']}: {entry['message']}\n"
    new_context = state["context"] + "\nCurrent Round History:\n" + current_round_history
    msg = st.session_state.agent2.negotiate(new_context)
    st.write(f"[Graph] Agent2 negotiation step {state['negotiation_step']}: {msg}")
    if msg.strip().lower() in ["take", "share"]:
        return {"agent2_final": msg.strip().lower()}
    else:
        log = state.get("negotiation_log", [])
        reasoning = state.get("agent2_reasoning", "")
        log.append({"agent": "Agent 2", "message": msg, "reasoning": reasoning})
        updated = {
            "negotiation_log": log,
            "negotiation_step": state["negotiation_step"] + 1,
            "current_turn": "agent1",
            "agent2_reasoning": ""
        }
        return updated

def negotiation_condition(state: RoundState):
    if state.get("agent1_final") or state.get("agent2_final"):
        return "finalize"
    if state["negotiation_step"] >= state["max_steps"]:
        return "finalize"
    return "dispatch"

def finalize(state: RoundState):
    current_round_history = ""
    for entry in state.get("negotiation_log", []):
        current_round_history += f"{entry['agent']}: {entry['message']}\n"
    new_context = state["context"] + "\nCurrent Round History:\n" + current_round_history
    if not state.get("agent1_final"):
        final1 = st.session_state.agent1.decide_final_move(new_context)
    else:
        final1 = state["agent1_final"]
    if not state.get("agent2_final"):
        final2 = st.session_state.agent2.decide_final_move(new_context)
    else:
        final2 = state["agent2_final"]
    return {"agent1_final": final1, "agent2_final": final2}

def end_node(state: RoundState):
    return state

def build_negotiation_graph(decision_first_mode: bool, private_reasoning: bool):
    graph_builder = StateGraph(RoundState)
    if decision_first_mode:
        graph_builder.add_node("agent1_pre_decision", agent1_pre_decision)
        graph_builder.add_node("agent2_pre_decision", agent2_pre_decision)
        graph_builder.add_edge(START, "agent1_pre_decision")
        graph_builder.add_edge("agent1_pre_decision", "agent2_pre_decision")
        graph_builder.add_edge("agent2_pre_decision", "choose_initial_turn")
    else:
        graph_builder.set_entry_point("choose_initial_turn")
    
    graph_builder.add_node("choose_initial_turn", choose_initial_turn)
    graph_builder.add_edge("choose_initial_turn", "dispatch")
    graph_builder.add_node("dispatch", dispatch)
    
    if private_reasoning:
        mapping = {"agent1_reasoning_node": "agent1_reasoning_node", "agent2_reasoning_node": "agent2_reasoning_node"}
    else:
        mapping = {"agent1_negotiate": "agent1_negotiate", "agent2_negotiate": "agent2_negotiate"}
    
    def condition_wrapper(state: RoundState):
        return choose_agent_node(state, private_reasoning)
    graph_builder.add_conditional_edges("dispatch", condition_wrapper, mapping)
    
    if private_reasoning:
        graph_builder.add_node("agent1_reasoning_node", agent1_reasoning_node)
        graph_builder.add_node("agent2_reasoning_node", agent2_reasoning_node)
        graph_builder.add_edge("agent1_reasoning_node", "agent1_negotiate")
        graph_builder.add_edge("agent2_reasoning_node", "agent2_negotiate")
    
    graph_builder.add_node("agent1_negotiate", agent1_negotiate)
    graph_builder.add_node("agent2_negotiate", agent2_negotiate)
    graph_builder.add_edge("agent1_negotiate", "check_final")
    graph_builder.add_edge("agent2_negotiate", "check_final")
    
    graph_builder.add_node("check_final", lambda state: state)
    graph_builder.add_conditional_edges("check_final", negotiation_condition, {"finalize": "finalize", "dispatch": "dispatch"})
    
    graph_builder.add_node("finalize", finalize)
    graph_builder.add_edge("finalize", END)
    
    graph_builder.add_node("end", end_node)
    graph_builder.set_finish_point("end")
    
    return graph_builder.compile()

# ------------------------------
# Streamlit App and Game Logic (UI and Persistent State)
# ------------------------------

# Create three tabs: Game, Settings, and Past Rounds.
tabs = st.tabs(["Game", "Settings", "Past Rounds"])

# Persistent state initialization.
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.0
if "llm_model" not in st.session_state:
    st.session_state.llm_model = build_llm("GPT4oMini", st.session_state.temperature if "temperature" in st.session_state else 0.0)
if "N_TURNS" not in st.session_state:
    st.session_state.N_TURNS = 10
if "reward_both_take" not in st.session_state:
    st.session_state.reward_both_take = 1
if "reward_take_vs_share" not in st.session_state:
    st.session_state.reward_take_vs_share = 10
if "reward_both_share" not in st.session_state:
    st.session_state.reward_both_share = 5
if "negotiation_rounds" not in st.session_state:
    st.session_state.negotiation_rounds = 5
if "decision_first_mode" not in st.session_state:
    st.session_state.decision_first_mode = False
if "private_reasoning" not in st.session_state:
    st.session_state.private_reasoning = False
if "full_dialogue" not in st.session_state:
    st.session_state.full_dialogue = ""  # Full conversation history from previous rounds.
if "round_dialogues" not in st.session_state:
    st.session_state.round_dialogues = {}  # Mapping: round number -> unified negotiation log.
if "round_final_moves" not in st.session_state:
    st.session_state.round_final_moves = {}  # Mapping: round number -> (agent1_move, agent2_move).

dummy_prompt = PromptTemplate(input_variables=["dummy"], template="{dummy}")

def init_agents():
    negotiation_chain_1 = LLMChain(llm=st.session_state.llm_model, prompt=dummy_prompt)
    decision_chain_1 = LLMChain(llm=st.session_state.llm_model, prompt=dummy_prompt)
    st.session_state.agent1 = NegotiationAgent("Agent 1", negotiation_chain_1, decision_chain_1)
    negotiation_chain_2 = LLMChain(llm=st.session_state.llm_model, prompt=dummy_prompt)
    decision_chain_2 = LLMChain(llm=st.session_state.llm_model, prompt=dummy_prompt)
    st.session_state.agent2 = NegotiationAgent("Agent 2", negotiation_chain_2, decision_chain_2)

if "agent1" not in st.session_state or "agent2" not in st.session_state:
    init_agents()

if "turn" not in st.session_state:
    st.session_state.turn = 1
if "your_points" not in st.session_state:
    st.session_state.your_points = 0
if "their_points" not in st.session_state:
    st.session_state.their_points = 0
if "history" not in st.session_state:
    st.session_state.history = []

game_state_placeholder = st.empty()
chat_placeholder = st.empty()
game_history_placeholder = st.empty()

def render_game_state():
    with game_state_placeholder.container():
        st.subheader(f"Turn {st.session_state.turn} of {st.session_state.N_TURNS}")
        st.write(f"**Agent 1 points:** {st.session_state.your_points}")
        st.write(f"**Agent 2 points:** {st.session_state.their_points}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Agent 1 Final Move")
            move_html = st.session_state.agent1_move if "agent1_move" in st.session_state else "Pending"
            st.markdown(render_emoji_square(move_html, "agent1"), unsafe_allow_html=True)
            st.write(move_html.capitalize())
        with col2:
            st.markdown("### Agent 2 Final Move")
            move_html = st.session_state.agent2_move if "agent2_move" in st.session_state else "Pending"
            st.markdown(render_emoji_square(move_html, "agent2"), unsafe_allow_html=True)
            st.write(move_html.capitalize())
    with game_history_placeholder.container():
        with st.expander("Game History", expanded=False):
            for line in st.session_state.history:
                st.write(line)

def render_negotiation_chat(negotiation_log):
    for entry in negotiation_log:
        agent = entry.get("agent")
        message = entry.get("message")
        reasoning = entry.get("reasoning", "")
        if agent == "Agent 1":
            st.chat_message("assistant").write(message)
        else:
            st.chat_message("user").write(message)
        if reasoning:
            st.markdown(f"<div style='margin-left: 20px; font-size: 0.8em; color: gray;'>Reasoning: {reasoning}</div>", unsafe_allow_html=True)

def render_final_moves(a1_move, a2_move):
    with st.chat_message("assistant"):
        st.markdown(f"<div class='agent1-text'>Final Move: {a1_move.capitalize()}</div>", unsafe_allow_html=True)
    with st.chat_message("user"):
        st.markdown(f"<div class='agent2-text'>Final Move: {a2_move.capitalize()}</div>", unsafe_allow_html=True)

def render_chat():
    with chat_placeholder.container(height=600):
        st.markdown("## Negotiation Chat (Current Round)")
        render_negotiation_chat(st.session_state.negotiation_log)
        st.markdown("## Final Moves")
        render_final_moves(st.session_state.agent1_move, st.session_state.agent2_move)

def render_past_round(round_num):
    if round_num in st.session_state.round_dialogues and round_num in st.session_state.round_final_moves:
        past_log = st.session_state.round_dialogues[round_num]
        a1_move, a2_move = st.session_state.round_final_moves[round_num]
        st.markdown(f"### Round {round_num}")
        render_negotiation_chat(past_log)
        render_final_moves(a1_move, a2_move)
    else:
        st.write("No data for that round.")

def skip_to_end():
    with st.spinner("Skipping to end..."):
        while st.session_state.turn <= st.session_state.N_TURNS:
            play_turn()

def play_turn():
    full_history = st.session_state.full_dialogue if st.session_state.full_dialogue else ""
    context = (
        f"Full Conversation History:\n{full_history}\n\n"
        f"Current Round (Turn {st.session_state.turn} of {st.session_state.N_TURNS}):\n"
        f"Agent 1 points: {st.session_state.your_points}. Agent 2 points: {st.session_state.their_points}. "
        f"Rewards: Both take: {st.session_state.reward_both_take} points each; "
        f"Take vs share: taker gets {st.session_state.reward_take_vs_share} points; "
        f"Both share: {st.session_state.reward_both_share} points each.\n"
        f"Game History Summary: {' | '.join(st.session_state.history)}"
    )
    
    round_state = {
        "context": context,
        "negotiation_log": [],
        "agent1_pre": "",
        "agent2_pre": "",
        "agent1_final": "",
        "agent2_final": "",
        "negotiation_step": 0,
        "max_steps": st.session_state.negotiation_rounds,
        "current_turn": "",
        "agent1_reasoning": "",
        "agent2_reasoning": ""
    }
    
    negotiation_graph = build_negotiation_graph(st.session_state.decision_first_mode, st.session_state.private_reasoning)
    final_state = negotiation_graph.invoke(round_state)
    
    a1_move = final_state.get("agent1_final", "take")
    a2_move = final_state.get("agent2_final", "take")
    
    st.session_state.agent1_move = a1_move
    st.session_state.agent2_move = a2_move
    st.session_state.negotiation_log = final_state.get("negotiation_log", [])
    
    round_num = st.session_state.turn
    st.session_state.round_dialogues[round_num] = st.session_state.negotiation_log.copy()
    st.session_state.round_final_moves[round_num] = (a1_move, a2_move)
    
    round_str = f"Round {round_num}:\n"
    for entry in st.session_state.negotiation_log:
        round_str += f"{entry['agent']}: {entry['message']}\n"
    round_str += f"Final Decision - Agent 1: {a1_move}, Agent 2: {a2_move}\n\n"
    st.session_state.full_dialogue += round_str
    
    st.write("[Console] Negotiation round completed with steps:", final_state.get("negotiation_step", 0))
    st.write("[Console] Agent 1 final move:", a1_move)
    st.write("[Console] Agent 2 final move:", a2_move)
    
    if a1_move == "take" and a2_move == "take":
        st.session_state.your_points += st.session_state.reward_both_take
        st.session_state.their_points += st.session_state.reward_both_take
    elif a1_move == "take" and a2_move == "share":
        st.session_state.your_points += st.session_state.reward_take_vs_share
    elif a1_move == "share" and a2_move == "take":
        st.session_state.their_points += st.session_state.reward_take_vs_share
    elif a1_move == "share" and a2_move == "share":
        st.session_state.your_points += st.session_state.reward_both_share
        st.session_state.their_points += st.session_state.reward_both_share
    else:
        st.warning("Unexpected move combination.")
    
    st.session_state.history.append(
        f"Turn {st.session_state.turn}: Agent 1 chose {a1_move}, Agent 2 chose {a2_move}"
    )
    st.session_state.turn += 1
    
    render_game_state()
    render_chat()

def game_tab():
    global game_state_placeholder, chat_placeholder, game_history_placeholder
    game_state_placeholder = st.empty()
    chat_placeholder = st.empty()
    game_history_placeholder = st.empty()
    
    render_game_state()
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("Play Next Turn"):
            if st.session_state.turn > st.session_state.N_TURNS:
                st.info("Game over! Reset to start again.")
            else:
                play_turn()
    with col_btn2:
        if st.button("Skip to End"):
            if st.session_state.turn > st.session_state.N_TURNS:
                st.info("Game already over.")
            else:
                skip_to_end()
    with col_btn3:
        if st.button("Reset Game"):
            st.session_state.turn = 1
            st.session_state.your_points = 0
            st.session_state.their_points = 0
            st.session_state.history = []
            st.session_state.full_dialogue = ""
            st.session_state.round_dialogues = {}
            st.session_state.round_final_moves = {}
            st.rerun()
    
    if st.session_state.turn > st.session_state.N_TURNS:
        st.subheader("Final Scores")
        st.write(f"**Agent 1 points:** {st.session_state.your_points}")
        st.write(f"**Agent 2 points:** {st.session_state.their_points}")

def settings_tab():
    st.header("Settings")
    model_choice = st.selectbox(
        "Select the LLM model to use:",
        options=list(available_models.keys()),
        index=0,
    )
    st.write("Selected model:", model_choice)
    st.session_state.selected_model = model_choice
    temperature = st.slider("Set Temperature", 0.0, 1.0, st.session_state.temperature, step=0.05)
    st.session_state.temperature = temperature
    st.session_state.llm_model = build_llm(model_choice, temperature)
    # Reinitialize agents with new LLM.
    negotiation_chain_1 = LLMChain(llm=st.session_state.llm_model, prompt=dummy_prompt)
    decision_chain_1 = LLMChain(llm=st.session_state.llm_model, prompt=dummy_prompt)
    st.session_state.agent1 = NegotiationAgent("Agent 1", negotiation_chain_1, decision_chain_1)
    negotiation_chain_2 = LLMChain(llm=st.session_state.llm_model, prompt=dummy_prompt)
    decision_chain_2 = LLMChain(llm=st.session_state.llm_model, prompt=dummy_prompt)
    st.session_state.agent2 = NegotiationAgent("Agent 2", negotiation_chain_2, decision_chain_2)
    
    num_turns = st.number_input("Set number of turns:", min_value=1, value=st.session_state.N_TURNS, step=1)
    st.session_state.N_TURNS = num_turns
    
    st.markdown("### Reward Settings")
    st.session_state.reward_both_take = st.number_input("Reward for both taking:", min_value=0, value=st.session_state.reward_both_take, step=1)
    st.session_state.reward_take_vs_share = st.number_input("Reward for taking when opponent shares:", min_value=0, value=st.session_state.reward_take_vs_share, step=1)
    st.session_state.reward_both_share = st.number_input("Reward for both sharing:", min_value=0, value=st.session_state.reward_both_share, step=1)
    
    st.markdown("### Negotiation Settings")
    st.session_state.negotiation_rounds = st.number_input("Number of negotiation rounds:", min_value=1, value=st.session_state.negotiation_rounds, step=1)
    
    st.markdown("### Game Mode")
    st.session_state.decision_first_mode = st.checkbox("Decision First Mode", value=st.session_state.decision_first_mode)
    
    st.markdown("### Private Reasoning")
    st.session_state.private_reasoning = st.checkbox("Enable Private Reasoning Step", value=st.session_state.private_reasoning)

def past_rounds_tab():
    st.header("Past Rounds")
    if st.session_state.round_dialogues:
        rounds = sorted(st.session_state.round_dialogues.keys())
        selected_round = st.selectbox("Select a round to view", rounds)
        render_past_round(selected_round)
    else:
        st.write("No rounds have been completed yet.")

with tabs[0]:
    game_tab()
with tabs[1]:
    settings_tab()
with tabs[2]:
    past_rounds_tab()
