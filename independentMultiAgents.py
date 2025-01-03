import os
from dotenv import load_dotenv
load_dotenv()


from crewai import Crew, Agent, Task, Process
from IPython.display import display, Markdown #can use to display markdown in a better format
from langchain_openai import ChatOpenAI
import requests

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")

#code to fetch news from using newsapi

def fetch_news(country="us", category=None):
    base_url = "https://newsapi.org/v2/top-headlines"
    params = {
        "country": country,
        "apiKey": news_api_key
    }
    # Add a category if provided
    if category:
        params["category"] = category
    # Make a GET request to the News API
    response = requests.get(base_url, params=params)
    data = response.json()
    # Check if the request was successful
    if response.status_code == 200 and 'articles' in data:
        # Extract the titles of the first 10 articles
        headlines = [article['title'] for article in data['articles'][:10]]
        return f"Here are the top news for {country.upper()} ({category or 'all categories'}):\n" + "\n".join(headlines)
    else:
        # Return an error message if the request failed
        return f"Sorry, I couldn't fetch the news at the moment: {data.get('message', 'Unknown error')}"


#an empty variable for userInput
user_input= "what is happening to tourist numbers in India?"


######
#Create AGENTS#

query_creator = Agent(
            role="SQL Query Creator",
            goal="You are supposed to create sql queries for the questions asked",
            backstory="You create sql queries based on requirements like what is total number of users in particular database",
            llm = ChatOpenAI(model="gpt-4o",temperature=0),
            verbose=True
        )

news_fetch_agent= Agent(
            role="News Reporter",
            goal="Analyze each news story and create detailed markdown summary",
            backstory="You analyze and write a report which is engaging for audience and easy to understand in markdown format",
            functions=[fetch_news],
            llm = ChatOpenAI(model="gpt-4o",temperature=0),
            verbose=True
        )


### Creating Tasks

query_creator_task = Task(
    description= "create a sql query for {user_input}",
    agent = query_creator,
    expected_output="The sql query is: ",
    verbose=True
)

#define writing task
writing_task = Task(
    description= "Generate a report based on the news in mardown summary for {user_input}",
    expected_output="a detailed and coherent report in Markdown ",
    agent=news_fetch_agent,
)


#Creating two crew that binds both agents

query_crew = Crew(
    tasks=[query_creator_task],
    agents=[query_creator],
    process=Process.sequential,
    verbose= True
)
news_crew = Crew(
    tasks=[writing_task],
    agents=[news_fetch_agent],
    process=Process.sequential,
)

### converting above crews to tools to be used in main crew

from crewai_tools import tool
@tool
def query_crew_tool() -> str:
    """Triggers crew2 for solving math problems."""
    return query_crew.kickoff({"user_input":user_input})

@tool
def news_crew_tool() -> str:
    """Triggers crew3 for fetching news and creating a report."""
    return news_crew.kickoff({"user_input":user_input})


decision_agent = Agent(
        role="Decision Maker",
        goal= "Analyze the task and decide if it is a sql query problem or news-fetching situation. user_input and delegate efficiently",
        backstory="An intelligent assistant capable of discerning the nature of tasks efficiently and delegating efficiently.",
        llm = ChatOpenAI(model="gpt-4o",temperature=0),
        tools=[query_crew_tool,news_crew_tool],
        verbose=True
    )
decision_task = Task(
    description=(
        "Analyze the input to determine if it is a sql query problem or a news-fetch task. user_input = {user_input}"
        "If it's a query problem, use the 'query_crew_tool' tool. If it's a news task, use the 'news_crew_tool' tool."
    ),
    expected_output="output from the tools:",
    agent=decision_agent,
)

crew1 = Crew(
    tasks=[decision_task],
    agents=[decision_agent],
    process=Process.sequential,
    verbose=True
)

while(user_input != "q"):
    user_input = str(input("Ask me for any news or if you want to create any queries for sql tasks give me in text format."))
    result=""
    if str(user_input)=="q":
        break
    else:
        try:
            result = crew1.kickoff({"user_input":user_input})
            print(f"\nResult is: \n{result}")
        except Exception as e:
            print(f"An error occurred {e}")
            
    