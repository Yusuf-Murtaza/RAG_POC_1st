"""Step 5. Wrap the retriever as a tool the agent can call"""

from langchain.tools import tool


def create_search_tool(reriever):
    """Return a @tool function that searches the HR policy documents"""
    @tool
    def search_hr_policy(question: str) -> str:
        """Search HR policy documents for a specific question related to leave, work from home, 
        probation, notice period, reimbursement, code of conduct, holidays, or exit process."""
        matching_chunks = reriever.invoke(question)
        return "\n\n".join([chunk.page_content for chunk in matching_chunks])

    return search_hr_policy



