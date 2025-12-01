"""
Orchestrator Agent
Routes user requests to appropriate agents and manages conversation flow
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import config  # Load environment variables
from agents.extractor_agent import extract_from_pdf, extract_from_text
from agents.interpreter_agent import interpret_lab_results, quick_question
from agents.general_qa_agent import ask_general_question, check_if_medical_question
from typing import Optional, Dict, Any
import json


class HealthReportOrchestrator:
    """
    Central orchestrator that coordinates all agents and manages conversation state.
    """

    def __init__(self):
        """Initialize orchestrator with empty state"""
        self.lab_reports = {}  # Store extracted lab reports by user/session
        self.conversation_history = []  # Track conversation flow

    async def process_pdf(self, pdf_path: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Process a PDF lab report.

        Args:
            pdf_path: Path to PDF file
            user_id: User identifier (for tracking multiple users)

        Returns:
            Dict with:
                - status: "success" or "error"
                - data: Structured lab data
                - summary: Human-readable summary
                - interpretation: Medical interpretation
        """
        print(f"\n{'='*60}")
        print("PROCESSING LAB REPORT PDF")
        print(f"{'='*60}\n")

        try:
            # Step 1: Extract structured data
            print("Step 1/3: Extracting data from PDF...")
            lab_data = await extract_from_pdf(pdf_path)

            # Store in memory
            self.lab_reports[user_id] = lab_data
            print("[OK] Lab data extracted and stored\n")

            # Step 2: Generate interpretation
            print("Step 2/3: Generating medical interpretation...")
            interpretation = await interpret_lab_results(lab_data)
            print("[OK] Interpretation complete\n")

            # Step 3: Create summary
            patient = lab_data.get('patient', {})
            num_tests = len(lab_data.get('tests', []))
            abnormal = len([t for t in lab_data.get('tests', [])
                          if t.get('flag') in ['HIGH', 'LOW', 'ABNORMAL']])

            summary = f"""
[Report processed for: {patient.get('name', 'Unknown')}]
   Tests: {num_tests} total, {abnormal} abnormal
   Status: Ready for questions

[You can now ask:]
   - "Why is my [test name] high/low?"
   - "What should I do about [test name]?"
   - "Explain my results in simple terms"
"""

            result = {
                "status": "success",
                "data": lab_data,
                "summary": summary,
                "interpretation": interpretation,
                "message": f"[OK] Report processed successfully! Found {num_tests} tests."
            }

            # Track in history
            self.conversation_history.append({
                "type": "pdf_upload",
                "user_id": user_id,
                "patient": patient.get('name'),
                "tests": num_tests
            })

            return result

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": f"[ERROR] Failed to process PDF: {str(e)}"
            }

    async def process_question(self, question: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Process a user question and route to appropriate agent.

        Args:
            question: User's question
            user_id: User identifier

        Returns:
            Dict with:
                - status: "success" or "error"
                - answer: The response
                - agent_used: Which agent provided the answer
                - context_used: Whether lab data was used
        """
        print(f"\n{'='*60}")
        print("PROCESSING QUESTION")
        print(f"{'='*60}")
        print(f"[Question: {question}]\n")

        try:
            # Check if we have lab data for this user
            lab_data = self.lab_reports.get(user_id)
            has_context = lab_data is not None

            if has_context:
                patient = lab_data.get('patient', {})
                print(f" Context available: {patient.get('name', 'Unknown')}'s lab report")
            else:
                print("No lab context available")

            # Classify question
            classification = await check_if_medical_question(question)
            is_medical = classification["is_medical"]

            print(f" Classification: {'Medical' if is_medical else 'General'}")

            # Route to appropriate agent
            if is_medical:
                print(f"→ Routing to: General QnA (with {'context' if has_context else 'no context'})\n")

                # For medical questions, use general QnA with context
                # (It will use lab data if available)
                answer = await ask_general_question(question, lab_data=lab_data)
                agent_used = "general_qa_with_context"

            else:
                print(f"→ Routing to: General QnA\n")

                # General questions
                answer = await ask_general_question(question, lab_data=lab_data)
                agent_used = "general_qa"

            result = {
                "status": "success",
                "answer": answer,
                "agent_used": agent_used,
                "context_used": has_context,
                "classification": classification
            }

            # Track in history
            self.conversation_history.append({
                "type": "question",
                "user_id": user_id,
                "question": question[:50],
                "agent": agent_used,
                "had_context": has_context
            })

            return result

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": f" Failed to process question: {str(e)}"
            }

    async def get_full_interpretation(self, user_id: str = "default") -> Dict[str, Any]:
        """
        Get the full medical interpretation for user's lab report.

        Args:
            user_id: User identifier

        Returns:
            Dict with interpretation or error
        """
        lab_data = self.lab_reports.get(user_id)

        if not lab_data:
            return {
                "status": "error",
                "message": " No lab report found. Please upload a report first."
            }

        try:
            print("\n[Generating full interpretation...]")
            interpretation = await interpret_lab_results(lab_data)

            return {
                "status": "success",
                "interpretation": interpretation
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": f" Failed to generate interpretation: {str(e)}"
            }

    def get_lab_summary(self, user_id: str = "default") -> Dict[str, Any]:
        """
        Get summary of stored lab data.

        Args:
            user_id: User identifier

        Returns:
            Dict with lab summary
        """
        lab_data = self.lab_reports.get(user_id)

        if not lab_data:
            return {
                "status": "error",
                "message": "No lab report available"
            }

        patient = lab_data.get('patient', {})
        tests = lab_data.get('tests', [])
        abnormal = [t for t in tests if t.get('flag') in ['HIGH', 'LOW', 'ABNORMAL']]

        return {
            "status": "success",
            "patient": patient.get('name'),
            "total_tests": len(tests),
            "abnormal_tests": len(abnormal),
            "abnormal_list": [
                {
                    "name": t.get('name'),
                    "value": t.get('result'),
                    "unit": t.get('unit'),
                    "flag": t.get('flag')
                }
                for t in abnormal
            ]
        }

    def clear_data(self, user_id: str = "default"):
        """Clear stored data for a user"""
        if user_id in self.lab_reports:
            del self.lab_reports[user_id]
        self.conversation_history = [h for h in self.conversation_history
                                    if h.get('user_id') != user_id]

    def get_conversation_history(self) -> list:
        """Get conversation history"""
        return self.conversation_history


# Global orchestrator instance
orchestrator = HealthReportOrchestrator()


if __name__ == "__main__":
    print("Orchestrator Module Loaded")
    print("Coordinates: Extractor, Interpreter, and General QnA agents")
