import json
import os
import requests
import time
import random
from typing import List
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("weather_service")

@mcp.tool()
def get_current_temperature(location: str, unit: str = "celsius") -> str:
    """Get current temperature at a location.

    Args:
        location (str): The location to get the temperature for.
        unit (str, optional): The unit of temperature to return. Defaults to "celsius".

    Returns:
        str: The current temperature at the specified location in the specified unit.
    """
    # Placeholder implementation
    temperature = 24
    return f"The current temperature in {location} is {temperature:.2f} {unit}."
@mcp.tool()
def get_temperature_date(location: str, date: str, unit: str = "celsius"):
    """Get temperature at a location and date.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        date: The date to get the temperature for, in the format "Year-Month-Day".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, the date and the unit in a dict
    """
    return {
        "temperature": 32,
        "location": location,
        "date": date,
        "unit": unit,
    }
if __name__ == "__main__":
    mcp.run(transport="stdio")


