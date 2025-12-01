"""
Orchestrator with Sessions & Memory
Adds persistent conversation tracking and long-term memory storage
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.genai import types
from agents.extractor_agent import extract_from_pdf, extractor_agent
from agents.interpreter_agent import interpreter_agent
from agents.general_qa_agent import general_qa_agent
from typing import Optional, Dict, Any
import json


class HealthReportOrchestratorWithMemory:
    """
    Enhanced orchestrator with session management and memory persistence.

    Key Features:
    - Session tracking (conversation threads)
    - Memory storage (long-term knowledge)
    - Auto-save callbacks
    - Multi-user support
    """

    def __init__(self):
        """Initialize with session and memory services"""
        # Services from Day 3
        self.session_service = InMemorySessionService()
        self.memory_service = InMemoryMemoryService()

        # Application identifier
        self.app_name = "health_report_assistant"

        # Local state
        self.lab_reports = {}  # Quick access to lab data

        print("[OK] Orchestrator initialized with Sessions & Memory")

    async def create_or_get_session(self, user_id: str, session_id: str = None):
        """
        Create or retrieve a session for a user.

        Args:
            user_id: User identifier
            session_id: Optional session ID (creates new if not provided)

        Returns:
            Session object
        """
        if session_id is None:
            session_id = f"session_{user_id}"

        try:
            session = await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )
        except:
            # Session already exists, retrieve it
            session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )

        return session

    async def process_pdf_with_memory(
        self,
        pdf_path: str,
        user_id: str = "default",
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Process PDF and store in both session and memory.

        Args:
            pdf_path: Path to PDF
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Result dict with extraction and interpretation
        """
        print(f"\n{'='*60}")
        print("PROCESSING PDF WITH MEMORY")
        print(f"{'='*60}\n")

        try:
            # Create/get session
            session = await self.create_or_get_session(user_id, session_id)
            print(f"[Session: {session.id}]")

            # Extract data
            print("Step 1/3: Extracting from PDF...")
            lab_data = await extract_from_pdf(pdf_path)

            # Store in local memory
            self.lab_reports[user_id] = lab_data
            print("[OK] Lab data extracted\n")

            # Step 2: Generate interpretation
            print("Step 2/3: Generating interpretation...")
            from agents.interpreter_agent import interpret_lab_results
            interpretation = await interpret_lab_results(lab_data)
            print("[OK] Interpretation complete\n")

            # Step 3: Save to memory
            print("Step 3/3: Saving to long-term memory...")

            # Refresh session to get latest events (from interpretation)
            session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session.id
            )

            # Save session to memory (includes all events)
            await self.memory_service.add_session_to_memory(session)
            print("[OK] Saved to memory\n")

            # Prepare summary data
            patient = lab_data.get('patient', {})
            abnormal_tests = [t for t in lab_data.get('tests', []) if t.get('flag') in ['HIGH', 'LOW']]

            # Summary
            summary = f"""
[REPORT PROCESSED]
   Patient: {patient.get('name', 'Unknown')}
   Tests: {len(lab_data.get('tests', []))} total, {len(abnormal_tests)} abnormal
   Session: {session.id}

[INFO] Your conversation history is now tracked!
   All interactions will be remembered.
"""

            return {
                "status": "success",
                "data": lab_data,
                "interpretation": interpretation,
                "summary": summary,
                "session_id": session.id
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": f"[ERROR] Failed: {str(e)}"
            }

    async def process_question_with_memory(
        self,
        question: str,
        user_id: str = "default",
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Process question with session context and memory.

        Args:
            question: User's question
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Answer with context
        """
        print(f"\n{'='*60}")
        print("PROCESSING QUESTION WITH MEMORY")
        print(f"{'='*60}")
        print(f"[QUESTION] {question}\n")

        try:
            # Get or create session
            session = await self.create_or_get_session(user_id, session_id)
            print(f"[Session: {session.id}]")

            # Check for lab data
            lab_data = self.lab_reports.get(user_id)
            has_lab_context = lab_data is not None

            if has_lab_context:
                patient = lab_data.get('patient', {})
                print(f"[Lab context: {patient.get('name', 'Unknown')}'s report]")

            # Search memory for relevant context
            print("[Searching memory...]")
            memory_results = await self.memory_service.search_memory(
                app_name=self.app_name,
                user_id=user_id,
                query=question
            )

            print(f"   Found {len(memory_results.memories)} relevant memories")

            # Create runner with memory
            runner = Runner(
                agent=general_qa_agent,
                app_name=self.app_name,
                session_service=self.session_service,
                memory_service=self.memory_service
            )

            # Build prompt with context
            from agents.general_qa_agent import ask_general_question
            answer = await ask_general_question(question, lab_data=lab_data)

            # Refresh session to capture the Q&A interaction
            session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session.id
            )

            # Auto-save to memory
            await self.memory_service.add_session_to_memory(session)

            print("[OK] Answer saved to session & memory\n")

            return {
                "status": "success",
                "answer": answer,
                "session_id": session.id,
                "memory_count": len(memory_results.memories),
                "context_used": has_lab_context
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": f"[ERROR] Failed: {str(e)}"
            }

    async def get_session_history(self, user_id: str, session_id: str = None) -> Dict[str, Any]:
        """
        Get conversation history from session.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Session history
        """
        try:
            if session_id is None:
                session_id = f"session_{user_id}"

            session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )

            messages = []
            for event in session.events:
                if event.content and event.content.parts:
                    text = event.content.parts[0].text
                    messages.append({
                        "role": event.content.role,
                        "text": text[:100] + "..." if len(text) > 100 else text
                    })

            return {
                "status": "success",
                "session_id": session.id,
                "message_count": len(messages),
                "messages": messages
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# Global instance
orchestrator_with_memory = HealthReportOrchestratorWithMemory()


if __name__ == "__main__":
    print("Orchestrator with Sessions & Memory loaded")
    print("Features: Session tracking, Memory storage, Auto-save")
