import os
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class Analyst:
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        llm = OpenAI(api_token=OPENAI_API_KEY)
        self.agent = SmartDataframe(df, config={"llm": llm})

    def run(self, inquiry):
        return self.agent.chat(inquiry)


if __name__ == "__main__":
    inquiry = "What is the amount spent per day? Plot the result."
    analyst = Analyst("data.csv")
    analyst.run(inquiry)
