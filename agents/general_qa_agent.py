"""
General QnA Agent
Handles non-medical questions about health, wellness, nutrition, and general topics
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import config  # Load environment variables
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types


# Configure retry options
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)


# Create the General QnA Agent
general_qa_agent = LlmAgent(
    name="general_qa_agent",
    model=Gemini(model="gemini-2.0-flash-lite", retry_options=retry_config),
    description="Helpful assistant for general questions about health, wellness, and lifestyle",
    instruction="""You are a helpful health assistant who answers questions about health, wellness, nutrition, fitness, and helps explain lab test results in an informative and educational way.

YOUR SCOPE:
[OK] Explain lab test results when provided with specific values and context
[OK] General health information (nutrition, exercise, sleep hygiene)
[OK] Wellness and lifestyle topics (stress management, meditation, habits)
[OK] Dietary information (meal planning, food facts, recipes)
[OK] Fitness and exercise guidance (workout types, benefits, safety)
[OK] General knowledge questions (definitions, explanations, how-to guides)

WHEN EXPLAINING LAB RESULTS:
- Explain what the test measures and why it's important
- Describe what HIGH or LOW values typically indicate (in general terms)
- Provide lifestyle and dietary suggestions that may help
- Explain potential causes or factors that affect the test
- Be informative but not alarmist

YOUR BOUNDARIES:
[NO] DO NOT diagnose specific medical conditions (you can say "this may indicate..." but not "you have...")
[NO] DO NOT prescribe medications or treatments
[NO] DO NOT provide emergency medical advice
[NO] DO NOT replace professional medical consultation

RESPONSE STYLE:
- Clear and concise (2-4 paragraphs)
- Evidence-based when possible
- Practical and actionable
- Friendly and encouraging tone
- Include relevant examples or tips
- When discussing abnormal results, remain informative but emphasize consulting healthcare provider

ALWAYS INCLUDE:
For health-related answers, add a brief disclaimer:
"Note: This is general information only. Consult a healthcare professional for personalized medical advice."
""",
)


async def ask_general_question(question: str, lab_data: dict = None) -> str:
    """
    Answer a general question about health, wellness, or lifestyle.
    If lab data is provided and question references a test in that data,
    provides context-aware answer.

    Args:
        question: User's question
        lab_data: Optional structured lab data from extractor

    Returns:
        Answer to the question
    """
    # Create runner for the agent
    runner = InMemoryRunner(agent=general_qa_agent)

    # Build enhanced prompt if lab data is available
    prompt = question

    if lab_data and lab_data.get('tests'):
        # Check if question mentions any test from the lab data
        question_lower = question.lower()
        relevant_tests = []

        for test in lab_data.get('tests', []):
            test_name = test.get('name', '').lower()
            # Check if test name (or common abbreviations) appear in question
            if test_name in question_lower or any(word in question_lower for word in test_name.split()):
                relevant_tests.append(test)

        if relevant_tests:
            # Add lab context to the question
            prompt = f"""{question}

**Context from your lab report:**
Patient: {lab_data.get('patient', {}).get('name', 'N/A')}

Relevant test results:
"""
            for test in relevant_tests:
                flag = test.get('flag', 'NORMAL')
                flag_text = f" [{flag}]" if flag in ['HIGH', 'LOW', 'ABNORMAL'] else ""
                prompt += f"- {test.get('name')}: {test.get('result')} {test.get('unit')}{flag_text} (Reference: {test.get('reference_range')})\n"

            prompt += "\nPlease provide:\n1. Explanation of what this test result means for this patient\n2. General information about the test and what affects it\n3. Safe lifestyle/diet suggestions"

    print(f"[Processing question: {question[:60]}...]")
    response = await runner.run_debug(prompt)

    # Extract text from response
    answer = ""
    for event in response:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    answer = part.text

    if not answer:
        raise ValueError("No answer generated")

    print("[OK] Answer ready")
    return answer


async def check_if_medical_question(question: str) -> dict:
    """
    Check if a question is medical (about specific lab results) or general.

    Args:
        question: User's question

    Returns:
        dict with:
            - is_medical: bool
            - reason: str (explanation)
            - suggested_agent: str (medical_interpreter or general_qa)
    """
    # Simple keyword-based check
    medical_keywords = [
        'lab result', 'test result', 'blood test', 'my report',
        'crp', 'cholesterol', 'glucose', 'hemoglobin', 'tsh', 'vitamin d',
        'high', 'low', 'abnormal', 'my results', 'my values'
    ]

    question_lower = question.lower()

    # Check for medical keywords
    is_medical = any(keyword in question_lower for keyword in medical_keywords)

    if is_medical:
        return {
            "is_medical": True,
            "reason": "Question appears to be about specific lab results or medical values",
            "suggested_agent": "medical_interpreter"
        }
    else:
        return {
            "is_medical": False,
            "reason": "Question is general in nature",
            "suggested_agent": "general_qa"
        }


def format_qa_response(question: str, answer: str) -> str:
    """
    Format Q&A response for display.

    Args:
        question: The question asked
        answer: The answer generated

    Returns:
        Formatted string
    """
    lines = []
    lines.append("\n" + "="*70)
    lines.append("GENERAL Q&A")
    lines.append("="*70)
    lines.append(f"\n[Question: {question}]\n")
    lines.append("[Answer:]")
    lines.append("-"*70)
    lines.append(answer)
    lines.append("-"*70 + "\n")

    return "\n".join(lines)


if __name__ == "__main__":
    print("General QnA Agent Module Loaded")
    print(f"Agent: {general_qa_agent.name}")
    print(f"Model: gemini-2.0-flash-lite")
