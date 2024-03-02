import os
# from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
# from langchain_commupipnity.chat_models import ChatOpenAI # deprecated
from langchain.chains import LLMChain
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

def ice_break(name: str) -> str:
    # load_dotenv()

    # print(os.environ['OPENAI_API_KEY'])

    # information = """
    # Elon Reeve Musk (born June 28, 1971) is a businessman and investor. He is the founder, chairman, CEO, and CTO of SpaceX; angel investor, CEO, product architect, and former chairman of Tesla, Inc.; owner, chairman, and CTO of X Corp.; founder of the Boring Company and xAI; co-founder of Neuralink and OpenAI; and president of the Musk Foundation. He is the second wealthiest person in the world, with an estimated net worth of US$232 billion as of December 2023, according to the Bloomberg Billionaires Index, and $182.6 billion according to Forbes, primarily from his ownership stakes in Tesla and SpaceX.

    # A member of the wealthy South African Musk family, Elon was born in Pretoria and briefly attended the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania, and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University, but dropped out after two days and, with his brother Kimbal, co-founded online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999, and, that same year Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal.
    # """

    linkedin_profile_url = linkedin_lookup_agent(name="Harrison Chase")

    summary_template = """
    Given the linkedin information {information} about a person I want you to create:
    1. Short summary.
    2. Two interesting facts about them.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # chain = chain.invoke(input={"information":information})
    # res = chain.invoke(input={"information":information})
    # print(res)

    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_profile_url  # "https://www.linkedin.com/in/harrison-chase-961287118/"
    )

    result = chain.run(information=linkedin_data)
    print(result)
    return result

if __name__ == "__main__":
    print("Hello LangChain!")
    result = ice_break(name="Harrison Chase")