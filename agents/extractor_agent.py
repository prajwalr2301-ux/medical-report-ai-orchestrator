"""
Health Report Extractor Agent

This module implements an AI agent that extracts structured data from medical lab reports.
It uses Google's Gemini model to parse unstructured PDF/text reports and convert them
into structured JSON format containing patient info, test results, and clinical data.

Key Features:
- Extracts patient demographics (name, DOB, ID, gender)
- Parses test results with values, units, and reference ranges
- Identifies abnormal results (HIGH, LOW, ABNORMAL flags)
- Groups tests by category (CBC, Metabolic Panel, etc.)
- Handles both PDF files and raw text input

Architecture:
- LLM-powered extraction using Gemini 2.0 Flash Lite
- Structured output with predefined JSON schema
- Robust error handling and retry logic
- Supports both PDF and raw text input
"""
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import config  # Load API keys and environment variables
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types
import json


# Configure retry options for API resilience
# - Retries up to 5 times for transient failures
# - Uses exponential backoff (base 7) starting at 1 second
# - Handles rate limiting (429) and server errors (500, 503, 504)
retry_config = types.HttpRetryOptions(
    attempts=5,           # Maximum retry attempts
    exp_base=7,          # Exponential backoff multiplier
    initial_delay=1,     # Start with 1 second delay
    http_status_codes=[429, 500, 503, 504]  # Retryable status codes
)


# Define the target JSON schema for extracted data
# This schema ensures consistent structured output from the LLM
# Used in the agent's instruction prompt to guide extraction format
EXTRACTION_SCHEMA = """
{
  "patient": {
    "name": "string",
    "dob": "string (MM/DD/YYYY)",
    "gender": "string",
    "patient_id": "string",
    "age": "number (optional)"
  },
  "clinic": {
    "name": "string",
    "address": "string",
    "phone": "string (optional)",
    "doctor": "string"
  },
  "report_info": {
    "collection_date": "string",
    "report_date": "string",
    "report_type": "string (e.g., 'Complete Blood Count', 'Metabolic Panel')"
  },
  "tests": [
    {
      "category": "string (e.g., 'CBC', 'Lipid Panel')",
      "name": "string (test name)",
      "result": "string or number",
      "unit": "string",
      "reference_range": "string",
      "flag": "string (NORMAL, HIGH, LOW, ABNORMAL, or null)"
    }
  ],
  "comments": "string (free text comments from the report)",
  "summary": "string (brief 2-3 sentence summary of key findings)"
}
"""


# Create the Extractor Agent using Google's Gemini model
# This is a specialized LLM agent configured to extract medical data
# Model choice: gemini-2.0-flash-lite for fast, cost-effective extraction
# Part of multi-agent healthcare analysis system
extractor_agent = LlmAgent(
    name="health_report_extractor",
    model=Gemini(
        model="gemini-2.0-flash-lite",  # Fast, lightweight Gemini model
        retry_options=retry_config       # Configured retry behavior
    ),
    description="Extracts structured medical data from lab reports",
    instruction=f"""You are a precise medical data extractor. Your job is to extract information from medical lab reports and structure it into valid JSON.

STRICT RULES:
1. Output ONLY valid JSON - no explanations, no markdown, no extra text
2. Follow this exact schema:
{EXTRACTION_SCHEMA}

3. For each test result:
   - Extract the exact test name
   - Extract the numeric result (or qualitative if applicable)
   - Extract the unit (e.g., g/dL, mg/dL, %)
   - Extract the reference range exactly as shown
   - Determine the flag: NORMAL, HIGH, LOW, or null if not specified

4. If a field is missing or unclear, use null
5. Group tests by category (CBC, Metabolic Panel, Lipid Panel, etc.)
6. Extract all comments verbatim
7. Create a brief summary highlighting abnormal values

IMPORTANT: Your output must be parseable by JSON.parse(). No extra formatting.
""",
)


async def extract_from_text(report_text: str) -> Dict[str, Any]:
    """
    Extract structured data from lab report text using the LLM agent.

    This function sends raw lab report text to the Gemini-powered extractor agent,
    which parses it and returns structured JSON data containing patient information,
    test results, and clinical findings.

    Args:
        report_text (str): Raw text extracted from PDF or input directly.
                          Can contain unstructured lab report data.

    Returns:
        Dict[str, Any]: Structured dictionary containing:
            - patient: Dict with name, DOB, gender, ID
            - clinic: Dict with clinic name, doctor, address
            - report_info: Dict with dates and report type
            - tests: List of test results with values, units, flags
            - comments: Free text comments from report
            - summary: AI-generated summary of key findings

    Raises:
        ValueError: If agent returns no response or invalid JSON
        json.JSONDecodeError: If response cannot be parsed as JSON

    Example:
        >>> text = "Patient: John Doe\\nCRP: 27.0 mg/L (High)"
        >>> data = await extract_from_text(text)
        >>> print(data['patient']['name'])
        'John Doe'
    """
    # Create runner for the agent (manages agent execution lifecycle)
    runner = InMemoryRunner(agent=extractor_agent)

    # Build the extraction prompt with clear instructions
    prompt = f"""Extract structured data from this medical lab report:

```
{report_text}
```

Return ONLY the JSON object, no other text."""

    # Run the agent asynchronously
    print("[Extracting structured data from report...]")
    response = await runner.run_debug(prompt)

    # Extract the JSON text from the agent's response
    # Response is a stream of events; we need the final text content
    json_text = None
    for event in response:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    json_text = part.text
                    break  # Found the text, exit inner loop

    # Validate that we got a response
    if not json_text:
        raise ValueError("No response from extractor agent")

    # Clean up the JSON response (LLMs sometimes wrap JSON in markdown)
    json_text = json_text.strip()
    if json_text.startswith('```json'):
        json_text = json_text[7:]  # Remove opening ```json
    if json_text.startswith('```'):
        json_text = json_text[3:]   # Remove opening ```
    if json_text.endswith('```'):
        json_text = json_text[:-3]  # Remove closing ```
    json_text = json_text.strip()

    # Parse and validate JSON structure
    try:
        structured_data = json.loads(json_text)
        print("[OK] Successfully extracted structured data")
        return structured_data
    except json.JSONDecodeError as e:
        # Log error details for debugging
        print(f"[ERROR] Failed to parse JSON: {e}")
        print(f"Raw response: {json_text[:500]}...")
        raise ValueError(f"Invalid JSON response from agent: {e}")


async def extract_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Extract structured data from a PDF lab report file.

    This is a convenience wrapper that combines PDF text extraction with
    the AI-powered data extraction. It handles the full pipeline from
    PDF file to structured medical data.

    Args:
        pdf_path (str): Absolute or relative path to the PDF file.
                       Example: "./reports/patient_lab_results.pdf"

    Returns:
        Dict[str, Any]: Same structured dictionary as extract_from_text().
                       Contains patient info, test results, and findings.

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If PDF cannot be read or data extraction fails

    Example:
        >>> data = await extract_from_pdf("./sample_report.pdf")
        >>> print(f"Found {len(data['tests'])} tests")
        Found 15 tests
    """
    # Import PDF utility (lazy import to avoid circular dependencies)
    from tools.pdf_utils import extract_text_from_pdf

    # Step 1: Extract raw text from PDF using PyPDF2 or pdfplumber
    print(f"[Reading PDF: {pdf_path}]")
    pdf_data = extract_text_from_pdf(pdf_path)
    print(f"   Pages: {pdf_data['page_count']}")
    print(f"   Method: {pdf_data['method']}")  # Shows which PDF library was used

    # Step 2: Send extracted text to AI agent for structured extraction
    return await extract_from_text(pdf_data['full_text'])


def format_extraction_summary(data: Dict[str, Any]) -> str:
    """
    Format the extracted data into a human-readable terminal summary.

    Converts the structured JSON data into a nicely formatted text summary
    suitable for console display, with sections for patient info, clinic,
    and test results grouped by category with abnormal flags highlighted.

    Args:
        data (Dict[str, Any]): Structured extraction result from extract_from_text()
                              or extract_from_pdf()

    Returns:
        str: Multi-line formatted summary with headers, sections, and test results.
            Includes visual indicators for abnormal values ([!] vs [OK])

    Example:
        >>> data = await extract_from_pdf("report.pdf")
        >>> summary = format_extraction_summary(data)
        >>> print(summary)
        ============================================================
        EXTRACTION SUMMARY
        ============================================================
        ...
    """
    lines = []
    lines.append("=" * 60)
    lines.append("EXTRACTION SUMMARY")
    lines.append("=" * 60)

    # Patient info
    if data.get('patient'):
        lines.append("\n👤 PATIENT")
        p = data['patient']
        lines.append(f"   Name: {p.get('name', 'N/A')}")
        lines.append(f"   DOB: {p.get('dob', 'N/A')}")
        lines.append(f"   Gender: {p.get('gender', 'N/A')}")
        lines.append(f"   ID: {p.get('patient_id', 'N/A')}")

    # Clinic info
    if data.get('clinic'):
        lines.append("\n CLINIC")
        c = data['clinic']
        lines.append(f"   {c.get('name', 'N/A')}")
        lines.append(f"   Doctor: {c.get('doctor', 'N/A')}")

    # Test summary
    if data.get('tests'):
        lines.append(f"\n TESTS ({len(data['tests'])} total)")
        abnormal_count = sum(1 for t in data['tests'] if t.get('flag') in ['HIGH', 'LOW', 'ABNORMAL'])
        lines.append(f"   Abnormal: {abnormal_count}")

        # Group by category
        categories = {}
        for test in data['tests']:
            cat = test.get('category', 'Other')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(test)

        for category, tests in categories.items():
            lines.append(f"\n   [{category}]")
            for test in tests:
                flag = test.get('flag', '')
                flag_icon = "[!] " if flag in ['HIGH', 'LOW', 'ABNORMAL'] else "[OK] "
                lines.append(f"      {flag_icon}{test.get('name')}: {test.get('result')} {test.get('unit')} [{flag or 'NORMAL'}]")

    # Summary
    if data.get('summary'):
        lines.append("\n[SUMMARY]")
        lines.append(f"   {data['summary']}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    print("Extractor Agent Module Loaded")
    print(f"Agent: {extractor_agent.name}")
    print(f"Model: gemini-2.0-flash-lite")
