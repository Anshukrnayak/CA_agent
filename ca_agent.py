import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()


class CAAgent:
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Define tools
        self.tools = [DuckDuckGoSearchRun()]

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful Chat Assistant. Answer questions helpfully and accurately.

            History: {history}
            Question: {input}
            """
        )

        # Create agent
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )

    def chat(self, input_text, history=None):
        response = self.agent_executor.invoke({
            "input": input_text,
            "history": history or []
        })
        return response["output"]


# CLI Interface
if __name__ == "__main__":
    agent = CAAgent()
    print("CA Agent: Hi! How can I help you today? (Type 'exit' to quit)")

    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = agent.chat(user_input, history)
        print(f"CA Agent: {response}")
        history.append(f"User: {user_input}\nAssistant: {response}")