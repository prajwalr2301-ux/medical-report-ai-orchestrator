"""
Medical Interpreter Agent

This module implements an AI agent that interprets medical lab results and explains them
in plain, accessible English. It provides context about test significance, potential
implications of abnormal values, and actionable lifestyle/dietary recommendations.

Key Features:
- Explains abnormal test results in patient-friendly language
- Provides context about what each test measures
- Offers safe, general diet and lifestyle suggestions
- Generates follow-up questions for doctor consultations
- Creates professional summaries for healthcare providers
- Includes mandatory medical disclaimers

Safety First:
- Never diagnoses conditions (uses "may indicate", "could suggest")
- Never prescribes medications or treatments
- Always emphasizes consulting healthcare professionals
- Empathetic, reassuring tone while maintaining accuracy

Architecture:
- Powered by Gemini 2.0 Flash Lite for fast, empathetic responses
- Structured output with predefined sections
- Context-aware interpretation based on patient data
- Medical disclaimer compliance for legal safety
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional

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


# Medical disclaimer template - required for all interpretations
# Ensures legal compliance and sets proper expectations
MEDICAL_DISCLAIMER = """
**MEDICAL DISCLAIMER**
This analysis is AI-generated for informational purposes only and is NOT medical advice.
Always consult a licensed healthcare professional for diagnosis, treatment, and medical decisions.
Do not make health decisions based solely on this information.
"""


# Create the Interpreter Agent using Google's Gemini model
# This agent specializes in translating medical jargon to patient-friendly language
# Model choice: gemini-2.0-flash-lite for fast, empathetic responses
# Part of multi-agent healthcare analysis system
interpreter_agent = LlmAgent(
    name="medical_interpreter",
    model=Gemini(
        model="gemini-2.0-flash-lite",  # Fast, lightweight Gemini model
        retry_options=retry_config       # Configured retry behavior
    ),
    description="Explains medical lab results in plain English with actionable insights",
    instruction="""You are a medical interpreter assistant who helps patients understand their lab results.

YOUR RESPONSIBILITIES:
1. Explain each abnormal test result in plain, simple English
2. Describe what the test measures and why it's important
3. Explain what HIGH or LOW values might indicate (general possibilities, not diagnosis)
4. Provide safe, general diet and lifestyle suggestions (NOT prescriptions)
5. Generate 3-5 relevant follow-up questions the patient should discuss with their doctor
6. Create a brief summary for sharing with doctors

STRICT SAFETY RULES:
- NEVER diagnose conditions (use words like "may indicate", "could suggest", "associated with")
- NEVER prescribe medications or specific treatments
- NEVER make urgent medical claims without context
- ALWAYS emphasize consulting a healthcare provider
- Keep explanations accessible to non-medical readers
- Be empathetic and reassuring while being accurate

OUTPUT FORMAT:
Provide a structured response with these sections:

## Test Results Overview
[Brief summary of total tests and how many are abnormal]

## Abnormal Results Explained
[For each abnormal test:]
- **Test Name**: What it is and why it's important
- **Your Result**: [value and flag]
- **What This Might Mean**: Simple explanation of implications
- **What You Can Do**: Safe diet/lifestyle suggestions

## Normal Results Summary
[Brief mention of tests that are in normal range]

## General Recommendations
[Overall diet and lifestyle suggestions based on all findings]

## Questions to Ask Your Doctor
[3-5 specific, relevant follow-up questions]

## Doctor's Summary
[Professional summary suitable for sharing with healthcare provider]

Remember: Be helpful, accurate, and always defer to medical professionals for diagnosis and treatment.
""",
)


async def interpret_lab_results(
    structured_data: Dict[str, Any],
    context: Optional[str] = None
) -> str:
    """
    Interpret structured lab results and provide plain English explanation.

    This function takes the structured JSON output from the extractor agent
    and generates a comprehensive, patient-friendly interpretation including
    explanations of abnormal values, dietary suggestions, and follow-up questions.

    Args:
        structured_data (Dict[str, Any]): JSON output from extractor_agent containing:
            - patient: Dict with patient demographics
            - tests: List of test results with values and flags
            - comments: Any clinical comments from the report
            - summary: Brief summary of key findings
        context (Optional[str]): Additional patient context if available.
            Example: "patient has type 2 diabetes", "pregnant", etc.

    Returns:
        str: Multi-section formatted interpretation including:
            - Overview of test results
            - Detailed explanations of abnormal findings
            - Diet and lifestyle recommendations
            - Questions for doctor
            - Professional summary
            - Medical disclaimer

    Raises:
        ValueError: If agent fails to generate an interpretation

    Example:
        >>> data = await extract_from_pdf("lab_report.pdf")
        >>> interpretation = await interpret_lab_results(data)
        >>> print(interpretation)
        ## Test Results Overview
        ...
    """
    # Create runner for the agent (manages agent execution lifecycle)
    runner = InMemoryRunner(agent=interpreter_agent)

    # Build comprehensive prompt with all available lab data
    # Include patient demographics for personalized interpretation
    prompt = f"""Interpret these lab results and provide a clear, helpful explanation:

**Patient Information:**
{json.dumps(structured_data.get('patient', {}), indent=2)}

**Test Results:**
{json.dumps(structured_data.get('tests', []), indent=2)}

**Report Comments:**
{structured_data.get('comments', 'None')}

**Extracted Summary:**
{structured_data.get('summary', 'None')}
"""

    # Add optional context if provided (e.g., medical history, medications)
    if context:
        prompt += f"\n**Additional Context:**\n{context}\n"

    prompt += "\nProvide a comprehensive interpretation following the structured format in your instructions."

    # Run the agent asynchronously
    print("[Interpreting lab results...]")
    response = await runner.run_debug(prompt)

    # Extract the interpretation text from agent's response
    # Response is a stream of events; we need the final text content
    interpretation = ""
    for event in response:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    interpretation = part.text

    # Validate that we got a response
    if not interpretation:
        raise ValueError("No interpretation generated")

    print("[OK] Interpretation complete")

    # Append mandatory medical disclaimer for legal compliance
    full_response = interpretation + "\n\n" + "="*60 + "\n" + MEDICAL_DISCLAIMER

    return full_response


async def quick_question(question: str, lab_data: Dict[str, Any]) -> str:
    """
    Answer a quick follow-up question about the lab results.

    Provides concise, context-aware answers to specific questions about
    lab results. Useful for chat interfaces where users ask targeted questions.

    Args:
        question (str): User's specific question about their lab results.
            Examples: "Why is my CRP high?", "What foods help reduce cholesterol?"
        lab_data (Dict[str, Any]): Structured lab data for context (from extractor)

    Returns:
        str: Brief answer (2-3 paragraphs) addressing the question, with
            context from the user's actual lab results and medical disclaimer.

    Raises:
        ValueError: If agent fails to generate an answer

    Example:
        >>> answer = await quick_question("Why is my CRP high?", lab_data)
        >>> print(answer)
        Your CRP level of 27.0 mg/L is elevated...
    """
    # Create runner for the agent
    runner = InMemoryRunner(agent=interpreter_agent)

    # Build targeted prompt with question and relevant lab data
    prompt = f"""User question: {question}

**Available Lab Data:**
{json.dumps(lab_data.get('tests', []), indent=2)}

Provide a brief, clear answer (2-3 paragraphs) to the user's question based on their lab results.
Include the medical disclaimer at the end.
"""

    # Run the agent asynchronously
    print(f"[Answering: {question}]")
    response = await runner.run_debug(prompt)

    # Extract answer text from response
    answer = ""
    for event in response:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    answer = part.text

    print("[OK] Answer ready")

    # Append medical disclaimer to answer
    return answer + "\n\n" + MEDICAL_DISCLAIMER


def format_for_print(interpretation: str) -> str:
    """
    Format interpretation for clean console/terminal output.

    Adds decorative borders and spacing for improved readability
    when displaying interpretations in command-line interfaces.

    Args:
        interpretation (str): The interpretation text to format

    Returns:
        str: Formatted string with header, borders, and proper spacing
            suitable for terminal display

    Example:
        >>> interp = await interpret_lab_results(data)
        >>> print(format_for_print(interp))
        ======================================================================
        MEDICAL LAB REPORT INTERPRETATION
        ======================================================================
        ...
    """
    lines = []
    lines.append("\n" + "="*70)
    lines.append("MEDICAL LAB REPORT INTERPRETATION")
    lines.append("="*70)
    lines.append(interpretation)
    lines.append("="*70 + "\n")

    return "\n".join(lines)


if __name__ == "__main__":
    # Module test/info output when run directly
    print("Medical Interpreter Agent Module Loaded")
    print(f"Agent: {interpreter_agent.name}")
    print(f"Model: gemini-2.0-flash-lite")
