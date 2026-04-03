import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
import os
from dotenv import load_dotenv

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo for current events or general questions."""
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=10,
        )
        data = resp.json()

        # Prefer the abstract if available
        if data.get("AbstractText"):
            return data["AbstractText"]

        # Fall back to related topics
        snippets = []
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and "Text" in topic:
                snippets.append(topic["Text"])

        return "\n".join(snippets) if snippets else f"No results found for: {query}"
    except Exception as e:
        return f"Search error: {e}"
    
## Arxiv and Wikipedia Tools
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

st.title("🔍 LangChain - Chat with search")

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key:",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant", "content":"Hi, I'm a chatbot who can search the web. How can i help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="What is machine learning"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key, model_name="qwen/qwen3-32b", streaming=True)
    tools=[web_search, arxiv, wiki]
    search_agent = create_agent(llm, tools)
    with st.chat_message("assistant"):
        ## Stream agent steps to show thoughts & actions (like StreamlitCallbackHandler)
        thoughts_container = st.container()
        response = ""
        for step in search_agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            stream_mode="updates",
        ):
            for node_name, node_output in step.items():
                messages = node_output.get("messages", [])
                for msg in messages:
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        ## Show the agent's action (tool call)
                        for tc in msg.tool_calls:
                            with thoughts_container.expander(
                                f"🔧 Calling **{tc['name']}**", expanded=False
                            ):
                                st.json(tc["args"])
                    elif isinstance(msg, ToolMessage):
                        ## Show the tool result
                        with thoughts_container.expander(
                            f"📄 Result from **{msg.name}**", expanded=False
                        ):
                            st.text(msg.content[:500])
                    elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                        ## Final answer
                        response = msg.content

        st.write(response)
        st.session_state.messages.append({"role":"assistant","content":response})