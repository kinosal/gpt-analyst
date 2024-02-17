import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import OpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class Analyst:
    def __init__(self, csv_file):
        self.agent = create_csv_agent(
            OpenAI(temperature=0),
            csv_file,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

    def run(self, inquiry):
        self.agent.invoke(inquiry)


if __name__ == "__main__":
    inquiry = "What is the amount spent per day?"
    analyst = Analyst("data.csv")
    analyst.run(inquiry)
