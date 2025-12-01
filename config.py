"""
Configuration Management for Health Report Assistant

This module handles application configuration loading from environment variables.
It loads API keys, cloud settings, and logging configuration from a .env file
and validates required credentials before application startup.

Environment Variables:
- GOOGLE_API_KEY (required): API key for Google Gemini models
- GOOGLE_CLOUD_LOCATION (optional): Cloud location, defaults to 'global'
- GOOGLE_GENAI_USE_VERTEXAI (optional): Whether to use Vertex AI (0 or 1)
- LOG_LEVEL (optional): Logging verbosity level, defaults to 'INFO'

Security:
- API keys are never logged in full (only first 20 and last 4 characters shown)
- Validates required credentials on import
- Raises ValueError if critical configuration is missing

Usage:
    from config import GOOGLE_API_KEY, LOG_LEVEL
    # Configuration is automatically loaded when this module is imported
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file in the project root
# This allows developers to keep sensitive credentials out of version control
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# ============================================================================
# API Configuration
# ============================================================================

# Google AI API key for Gemini model access
# Required for all agent functionality
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Google Cloud location for API requests
# Defaults to 'global' for worldwide access
GOOGLE_CLOUD_LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION', 'global')

# Flag to determine whether to use Vertex AI or AI Studio
# 0 = AI Studio (default), 1 = Vertex AI
GOOGLE_GENAI_USE_VERTEXAI = int(os.getenv('GOOGLE_GENAI_USE_VERTEXAI', '0'))

# ============================================================================
# Logging Configuration
# ============================================================================

# Logging level for application-wide logging
# Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# ============================================================================
# Validation
# ============================================================================

# Validate that required configuration is present
# Fail fast if critical credentials are missing
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in environment. "
        "Please create a .env file with your API key. "
        "See README.md for setup instructions."
    )

# ============================================================================
# Environment Variable Setup for ADK
# ============================================================================

# Set environment variables for Agent Development Kit (ADK) consumption
# ADK libraries read these environment variables directly
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
os.environ['GOOGLE_CLOUD_LOCATION'] = GOOGLE_CLOUD_LOCATION
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = str(GOOGLE_GENAI_USE_VERTEXAI)

# ============================================================================
# Startup Confirmation
# ============================================================================

# Log successful configuration load (with API key partially masked for security)
print("[OK] Configuration loaded successfully")
print(f"   API Key: {GOOGLE_API_KEY[:20]}...{GOOGLE_API_KEY[-4:]}")
print(f"   Location: {GOOGLE_CLOUD_LOCATION}")
print(f"   Use Vertex AI: {bool(GOOGLE_GENAI_USE_VERTEXAI)}")
