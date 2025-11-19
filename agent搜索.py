from google.adk.agents import Agent
from google.adk.tools import VertexAiSearchTool

# Configuration
# Replace with your Vertex AI Search Datastore ID, and respective region (e.g. us-central1 or global).
# Format: projects/gen-lang-client-0745685302/locations/global/collections/default_collection/dataStores/alphabet_1760802626649
DATASTORE_ID = "projects/gen-lang-client-0745685302/locations/global/collections/default_collection/dataStores/alphabet_1760802626649"

root_agent = Agent(
    name="vertex_search_agent",
    model="gemini-2.5-flash",
    instruction="Answer questions using Vertex AI Search to find information from internal documents. Always cite sources when available.",
    description="Enterprise document search assistant with Vertex AI Search capabilities",
    tools=[VertexAiSearchTool(data_store_id=DATASTORE_ID)]
)