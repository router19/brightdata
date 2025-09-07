# YouTube Tutorial Link - https://youtu.be/NF2aRqIlYNE

import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

load_dotenv()

async def run_agent():

    client = MultiServerMCPClient(
        {
            "bright_data": {
                "command": "npx",
                "args": ["@brightdata/mcp"],
                "env": {
                    "API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN"),
                    "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE", "unblocker"),
                    "BROWSER_ZONE": os.getenv("BROWSER_ZONE", "scraping_browser")
                },
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    model = init_chat_model(model="openai:gpt-4.1", api_key = os.getenv("OPENAI_API_KEY"))
    agent = create_react_agent(model, tools, prompt="You are a web search agent with access to brightdata tool to get latest data")
    message = """
        Provide comprehensive insights on Atorvastatin Calcium. Include information from all approved dosage forms.  For each section (Brand Name, Indications, etc.), ONLY extract verbatim text as found on US FDA https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm; EMA https://www.ema.europa.eu/en/homepage; https://www.medicines.org.uk/emc; TGA https://www.tga.gov.au/; Health Canada https://health-products.canada.ca/dpd-bdpp/.

        Do NOT use any other database or FDA page. Do not summarize. 
    
        Copy the content directly from the site and reference the page or section label. Use the most recent information available on the above sites. Highlight if there are any differences
        Brand name, Indications and Usage, Dosage and Administration, Dosage Forms and Strengths, Drug molecule details (structure, molecular weight, solubility, BCS class, mechanism of action, properties), Inactive ingredients, Mechanism of Action, Pharmacokinetics- Absorption, Distribution, Metabolism, Elimination and Excretion, Packaging and Storage  
        Extract information from the Orange Book (approval dates, RLD, RS, patent). ONLY extract the exact text as found on https://www.accessdata.fda.gov/scripts/cder/ob/index.cfm  in original tabular form. 
        Extract the information from the FDA dissolution database https://www.accessdata.fda.gov/scripts/cder/dissolution/index.cfm in original tabular form. """
    agent_response = await agent.ainvoke({"messages": message})
    print(agent_response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(run_agent())