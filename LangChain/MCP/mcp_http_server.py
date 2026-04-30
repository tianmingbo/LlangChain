from mcp.server.fastmcp import FastMCP

# Stateless + JSON response makes local debugging easier.
mcp = FastMCP("demo-http-server", stateless_http=True, json_response=True)


@mcp.tool()
def add(a: int, b: int) -> dict:
    """Add two numbers."""
    return {"sum": a + b}


@mcp.resource("note://http-demo")
def read_http_note() -> str:
    """Return a simple resource text."""
    return "This is an MCP HTTP demo resource: includes tool, resource, and prompt examples."


@mcp.prompt()
def weather_brief_prompt(city: str, date: str) -> str:
    """Build a weather briefing prompt template."""
    return (
        f"Please write a short weather briefing for {city} on {date}. "
        "Use a warm tone, keep it under 40 words, and end with one practical tip."
    )


if __name__ == "__main__":
    # Default endpoint: http://127.0.0.1:8000/mcp
    mcp.run(transport="streamable-http")
