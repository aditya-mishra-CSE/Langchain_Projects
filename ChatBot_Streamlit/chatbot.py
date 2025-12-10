from typing import Annotated

# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.getenv("TAVILY_API_KEY")
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# tool = TavilySearchResults(max_results=2)
tool = TavilySearch(max_results=2)
tools = [tool]

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile(checkpointer = memory)


# -----------------------------------

import streamlit as st

st.title("Chatbot App")

if "messages" not in st.session_state:
    st.session_state.messages = [] 

def stream_graph_updates(user_input: str):
    # st.session_state.messages.append(("user", user_input))

    assistant_response = ""

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # Stream from compiled graph
        for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, {"configurable": {"thread_id": "1"}}):
            # event is the current state dict, e.g. {"messages": [...]}
            for value in event.values():
                new_text = value["messages"][-1].content
                assistant_response += new_text
                message_placeholder.markdown(assistant_response)

    # Save assistant message
    st.session_state.messages.append(("assistant", assistant_response))

# Display previous chat history
for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(message)


if prompt := st.chat_input("What is your question?"):
    #Append user input to session state
    st.session_state.messages.append(("user",prompt))
    #Display user input as a chat message
    with st.chat_message("user"):
        st.markdown(prompt)
    

    #Get response from the chatbot based on user input
    response = stream_graph_updates(prompt)



