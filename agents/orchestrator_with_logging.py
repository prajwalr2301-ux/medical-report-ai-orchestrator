"""
Health Report Orchestrator with Observability

This module coordinates all AI agents in the health report analysis system and provides
comprehensive observability through logging, metrics, and session management.

Architecture:
- Orchestrates three specialized agents: Extractor, Interpreter, and General Q&A
- Manages user sessions for conversation continuity
- Tracks conversation history with memory services
- Records performance metrics (extraction time, interpretation time, etc.)
- Provides structured logging for debugging and monitoring

Components:
- Session Management: Tracks user interactions across requests
- Memory Service: Stores and retrieves conversation history
- Metrics Tracker: Records performance and usage statistics
- Logging: Detailed event logging for production monitoring

Production Features:
- Error handling and graceful degradation
- Performance timing for all operations
- Structured logging for log aggregation
- Metrics collection for SLA monitoring
"""
import sys
from pathlib import Path
import time
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import config
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.plugins import LoggingPlugin
from agents.extractor_agent import extract_from_pdf
from agents.interpreter_agent import interpret_lab_results
from agents.general_qa_agent import ask_general_question
from utils.logging_config import setup_logging, get_logger, metrics_tracker


class HealthReportOrchestratorWithLogging:
    """
    Main orchestrator that coordinates all agents with full observability.

    This class serves as the central coordinator for the health report analysis system.
    It manages the workflow between multiple specialized agents, handles session state,
    tracks conversation memory, and provides comprehensive logging and metrics.

    Key Responsibilities:
    - PDF Processing: Coordinates extraction and interpretation of lab reports
    - Q&A Handling: Routes health questions to appropriate agents
    - Session Management: Maintains user session state across requests
    - Memory Management: Stores conversation history for context
    - Observability: Logs all operations and tracks performance metrics
    - Error Handling: Gracefully handles failures with detailed error reporting

    Attributes:
        logger: Structured logger for operation tracking
        session_service: Manages user sessions (InMemorySessionService)
        memory_service: Stores conversation history (InMemoryMemoryService)
        app_name: Application identifier for session/memory services
        lab_reports: In-memory cache of processed lab reports by user_id
        metrics: Performance metrics tracker

    Example:
        >>> orchestrator = HealthReportOrchestratorWithLogging(log_level="INFO")
        >>> result = await orchestrator.process_pdf_with_logging(
        ...     pdf_path="report.pdf",
        ...     user_id="user123"
        ... )
        >>> print(result['status'])
        'success'
    """

    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the orchestrator with logging and observability components.

        Args:
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                           Defaults to "INFO". Use "DEBUG" for detailed tracing.

        Raises:
            ValueError: If log_level is invalid
        """
        # Setup logging
        self.logger = setup_logging(level=log_level)
        self.logger.info("="*60)
        self.logger.info("Initializing Health Report Orchestrator with Logging")
        self.logger.info("="*60)

        # Initialize session management service
        # InMemorySessionService: Lightweight in-memory session tracking
        # For production: consider persistent storage (Redis, Database)
        self.session_service = InMemorySessionService()

        # Initialize memory service for conversation history
        # InMemoryMemoryService: Stores conversation context
        # For production: consider vector database for semantic search
        self.memory_service = InMemoryMemoryService()

        # Application identifier for session/memory namespacing
        self.app_name = "health_report_assistant"

        # Local state: In-memory cache of processed lab reports
        # Key: user_id, Value: structured lab data
        # Note: This cache is session-scoped, not persistent
        self.lab_reports = {}

        # Metrics tracker for performance monitoring
        # Tracks: extraction time, interpretation time, errors, etc.
        self.metrics = metrics_tracker

        self.logger.info("Orchestrator initialized successfully")
        self.logger.debug(f"Session service: {type(self.session_service).__name__}")
        self.logger.debug(f"Memory service: {type(self.memory_service).__name__}")

    async def create_or_get_session(
        self,
        user_id: str,
        session_id: Optional[str] = None
    ):
        """
        Create a new session or retrieve an existing one for the user.

        Sessions provide continuity across multiple interactions with the system.
        Each session tracks conversation history, context, and state for a specific user.

        Args:
            user_id (str): Unique identifier for the user
            session_id (Optional[str]): Specific session ID to retrieve.
                                       If None, generates session ID from user_id.

        Returns:
            Session: ADK Session object with id, user_id, and state

        Example:
            >>> session = await orchestrator.create_or_get_session("user123")
            >>> print(session.id)
            'session_user123'
        """
        if session_id is None:
            session_id = f"session_{user_id}"

        self.logger.debug(f"Creating/retrieving session for user={user_id}, session={session_id}")

        try:
            session = await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )
            self.logger.info(f"Created new session: {session.id}")
        except:
            session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )
            self.logger.info(f"Retrieved existing session: {session.id}")

        return session

    async def process_pdf_with_logging(
        self,
        pdf_path: str,
        user_id: str = "default",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a medical lab report PDF with full observability and metrics tracking.

        This method orchestrates the complete PDF processing workflow:
        1. Creates/retrieves user session for tracking
        2. Extracts structured data from PDF using extractor agent
        3. Generates patient-friendly interpretation using interpreter agent
        4. Saves conversation to memory for future reference
        5. Tracks performance metrics for all operations

        Args:
            pdf_path (str): Path to the PDF lab report file
            user_id (str): Unique user identifier. Defaults to "default"
            session_id (Optional[str]): Specific session ID. If None, auto-generated.

        Returns:
            Dict[str, Any]: Processing result containing:
                - status (str): "success" or "error"
                - data (Dict): Structured lab data (on success)
                - interpretation (str): Patient-friendly explanation (on success)
                - summary (str): Brief summary with key findings (on success)
                - session_id (str): Session identifier for this interaction
                - metrics (Dict): Performance metrics (extraction_time, etc.)
                - error (str): Error message (on failure)
                - message (str): User-friendly error description (on failure)

        Example:
            >>> result = await orchestrator.process_pdf_with_logging(
            ...     pdf_path="lab_report.pdf",
            ...     user_id="patient_123"
            ... )
            >>> if result['status'] == 'success':
            ...     print(f"Found {len(result['data']['tests'])} tests")
            ...     print(result['interpretation'])
        """
        start_time = time.time()
        self.logger.info("="*60)
        self.logger.info(f"PROCESSING PDF - User: {user_id}")
        self.logger.info("="*60)
        self.logger.debug(f"PDF path: {pdf_path}")

        try:
            # Create/get session
            session = await self.create_or_get_session(user_id, session_id)

            # Step 1: Extract data
            self.logger.info("Step 1/3: Extracting from PDF...")
            extract_start = time.time()

            lab_data = await extract_from_pdf(pdf_path)

            extract_time = time.time() - extract_start
            self.metrics.record("pdf_extraction_time", extract_time, {"user_id": user_id})

            self.lab_reports[user_id] = lab_data
            num_tests = len(lab_data.get('tests', []))
            self.logger.info(f"Extracted {num_tests} tests in {extract_time:.2f}s")
            self.logger.debug(f"Patient: {lab_data.get('patient', {}).get('name')}")

            # Step 2: Generate interpretation
            self.logger.info("Step 2/3: Generating interpretation...")
            interpret_start = time.time()

            interpretation = await interpret_lab_results(lab_data)

            interpret_time = time.time() - interpret_start
            self.metrics.record("interpretation_time", interpret_time, {"user_id": user_id})
            self.logger.info(f"Interpretation complete in {interpret_time:.2f}s")

            # Step 3: Save to memory
            self.logger.info("Step 3/3: Saving to memory...")
            session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session.id
            )
            await self.memory_service.add_session_to_memory(session)
            self.logger.debug("Session saved to memory service")

            # Summary
            patient = lab_data.get('patient', {})
            abnormal_tests = [t for t in lab_data.get('tests', []) if t.get('flag') in ['HIGH', 'LOW']]

            total_time = time.time() - start_time
            self.metrics.record("pdf_processing_total_time", total_time, {"user_id": user_id})

            self.logger.info("="*60)
            self.logger.info(f"PDF Processing Complete - Total time: {total_time:.2f}s")
            self.logger.info(f"Patient: {patient.get('name')}, Tests: {num_tests}, Abnormal: {len(abnormal_tests)}")
            self.logger.info("="*60)

            summary = f"""
[REPORT PROCESSED]
   Patient: {patient.get('name', 'Unknown')}
   Tests: {num_tests} total, {len(abnormal_tests)} abnormal
   Session: {session.id}
   Processing time: {total_time:.2f}s

[INFO] Your conversation history is now tracked!
"""

            return {
                "status": "success",
                "data": lab_data,
                "interpretation": interpretation,
                "summary": summary,
                "session_id": session.id,
                "metrics": {
                    "extraction_time": extract_time,
                    "interpretation_time": interpret_time,
                    "total_time": total_time
                }
            }

        except Exception as e:
            self.logger.error(f"PDF processing failed: {str(e)}", exc_info=True)
            self.metrics.record("pdf_processing_errors", 1, {"user_id": user_id})
            return {
                "status": "error",
                "error": str(e),
                "message": f"[ERROR] Failed: {str(e)}"
            }

    async def process_question_with_logging(
        self,
        question: str,
        user_id: str = "default",
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Process question with detailed logging and metrics.
        """
        start_time = time.time()
        self.logger.info("="*60)
        self.logger.info(f"PROCESSING QUESTION - User: {user_id}")
        self.logger.info("="*60)
        self.logger.debug(f"Question: {question}")

        try:
            # Get session
            session = await self.create_or_get_session(user_id, session_id)

            # Check for lab data
            lab_data = self.lab_reports.get(user_id)
            has_lab_context = lab_data is not None

            if has_lab_context:
                patient = lab_data.get('patient', {})
                self.logger.info(f"Lab context available: {patient.get('name')}'s report")
                self.logger.debug(f"Tests in context: {len(lab_data.get('tests', []))}")
            else:
                self.logger.warning("No lab context available for this user")

            # Search memory
            self.logger.debug("Searching memory for relevant context...")
            memory_results = await self.memory_service.search_memory(
                app_name=self.app_name,
                user_id=user_id,
                query=question
            )
            self.logger.info(f"Found {len(memory_results.memories)} relevant memories")

            # Generate answer
            self.logger.info("Generating answer...")
            answer_start = time.time()

            answer = await ask_general_question(question, lab_data=lab_data)

            answer_time = time.time() - answer_start
            self.metrics.record("question_answer_time", answer_time, {"user_id": user_id})
            self.logger.info(f"Answer generated in {answer_time:.2f}s")

            # Save to memory
            session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session.id
            )
            await self.memory_service.add_session_to_memory(session)
            self.logger.debug("Session updated in memory")

            total_time = time.time() - start_time
            self.logger.info(f"Question processing complete in {total_time:.2f}s")

            return {
                "status": "success",
                "answer": answer,
                "session_id": session.id,
                "memory_count": len(memory_results.memories),
                "context_used": has_lab_context,
                "metrics": {
                    "answer_time": answer_time,
                    "total_time": total_time
                }
            }

        except Exception as e:
            self.logger.error(f"Question processing failed: {str(e)}", exc_info=True)
            self.metrics.record("question_processing_errors", 1, {"user_id": user_id})
            return {
                "status": "error",
                "error": str(e),
                "message": f"[ERROR] Failed: {str(e)}"
            }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics"""
        summary = self.metrics.get_summary()
        self.logger.info("Metrics Summary:")
        for metric_name, stats in summary.items():
            self.logger.info(f"  {metric_name}: avg={stats['average']:.2f}s, min={stats['min']:.2f}s, max={stats['max']:.2f}s, count={stats['count']}")
        return summary


# Global instance
orchestrator_with_logging = HealthReportOrchestratorWithLogging()


if __name__ == "__main__":
    print("Orchestrator with Logging loaded")
    print("Features: Sessions, Memory, Detailed Logging, Metrics Tracking")
