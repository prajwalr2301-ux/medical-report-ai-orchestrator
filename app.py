"""
Health Report Assistant - Web Application

A production-ready web interface for AI-powered medical lab report analysis.
Built with Streamlit for interactive report processing, interpretation, and
patient Q&A functionality.

Features:
- PDF lab report upload and processing
- AI-powered data extraction from unstructured PDFs
- Medical interpretation in patient-friendly language
- Interactive chat for health questions
- Session management and conversation history
- Real-time metrics and performance tracking

Architecture:
- Frontend: Streamlit web framework
- Backend: Multi-agent AI system (Extractor, Interpreter, Q&A)
- AI Models: Google Gemini 2.0 Flash Lite
- Deployment: Can run locally or on cloud platforms (Streamlit Cloud, GCP, AWS)

Usage:
    streamlit run app.py

Security & Compliance:
- Medical disclaimers on all AI-generated content
- No diagnosis or prescription capabilities
- Privacy-focused (in-memory processing)
- Secure API key management via environment variables
"""
import streamlit as st
import asyncio
import sys
from pathlib import Path
import pandas as pd
import time
from datetime import datetime

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

from agents.orchestrator_with_logging import HealthReportOrchestratorWithLogging

# Page configuration
st.set_page_config(
    page_title="Health Report Assistant",
    page_icon="utils/648c6d3f-de6a-4d7a-871e-11684521f749.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean styling
st.markdown("""
<style>
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }

    /* Card styling */
    .info-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        border-left: 3px solid #2196f3;
    }

    .warning-card {
        background: #fff3e0;
        border-left-color: #ff9800;
    }

    .danger-card {
        background: #ffebee;
        border-left-color: #f44336;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
        padding: 1.25rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }

    .metric-card h3 {
        margin: 0;
        font-size: 1.75rem;
        font-weight: 600;
    }

    .metric-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.9rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = HealthReportOrchestratorWithLogging(log_level="INFO")
    st.session_state.user_id = "web_user_001"
    st.session_state.lab_data = None
    st.session_state.interpretation = None
    st.session_state.chat_history = []
    st.session_state.session_id = f"web_session_{int(time.time())}"

# ============================================================================
# Async Execution Helper
# ============================================================================

def run_async(coro):
    """
    Execute an async coroutine in Streamlit's synchronous context.

    Streamlit runs in a synchronous environment, but our AI agents use
    async/await for non-blocking I/O. This helper bridges the gap.

    Args:
        coro: Async coroutine to execute

    Returns:
        Result from the coroutine

    Note:
        - Reuses event loop across calls to prevent "Event loop is closed" errors
        - Allows pending cleanup tasks to complete before returning
        - Essential for Streamlit compatibility with async agent code
    """
    try:
        # Try to get the existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        # Create a new event loop if none exists or it's closed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Run the coroutine and return result
    result = loop.run_until_complete(coro)

    # Don't close the loop immediately - let pending cleanup tasks complete
    # The loop will be reused for subsequent calls
    return result

# Header
st.markdown("""
<div class="main-header">
    <h1>Health Report Assistant</h1>
    <p>AI-Powered Lab Report Analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Upload Lab Report")

    uploaded_file = st.file_uploader(
        "Choose PDF file",
        type=['pdf'],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Save uploaded file
        pdf_path = Path("temp_upload.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Analyze Report", width='stretch', type="primary"):
            with st.spinner("Analyzing your lab report..."):
                progress_bar = st.progress(0)

                # Process PDF
                progress_bar.progress(30)
                result = run_async(
                    st.session_state.orchestrator.process_pdf_with_logging(
                        str(pdf_path),
                        user_id=st.session_state.user_id,
                        session_id=st.session_state.session_id
                    )
                )

                progress_bar.progress(100)

                if result["status"] == "success":
                    st.session_state.lab_data = result["data"]
                    st.session_state.interpretation = result["interpretation"]
                    st.session_state.chat_history = []  # Clear chat history for new report
                    st.success("Report analyzed successfully")
                else:
                    st.error(f"Error: {result['message']}")

    st.markdown("---")
    st.markdown("### About")
    st.caption("AI-powered analysis of lab results with health insights and dietary recommendations.")

# Main content area
if st.session_state.lab_data is None:
    # Welcome screen
    st.info("Upload a lab report PDF using the sidebar to get started")

else:
    # Report loaded - show tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Medical Interpretation",
        "Summary",
        "Lab Results",
        "Chat"
    ])

    with tab1:
        if st.session_state.interpretation:
            st.markdown(st.session_state.interpretation)
        else:
            st.info("No interpretation available")

    with tab2:

        # Summary metrics
        tests = st.session_state.lab_data.get('tests', [])
        total_tests = len(tests)
        abnormal_tests = len([t for t in tests if t.get('flag') in ['HIGH', 'LOW', 'ABNORMAL', 'INSUFFICIENT']])
        normal_tests = total_tests - abnormal_tests

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{total_tests}</h3>
                <p>Total Tests</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            color = "#4caf50" if abnormal_tests == 0 else "#ff9800" if abnormal_tests <= 3 else "#f44336"
            st.markdown(f"""
            <div class="metric-card" style="background: {color};">
                <h3>{abnormal_tests}</h3>
                <p>Abnormal Results</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                <h3>{normal_tests}</h3>
                <p>Normal Results</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Abnormal results highlight
        if abnormal_tests > 0:
            st.markdown("### Tests Requiring Attention")

            abnormal = [t for t in tests if t.get('flag') in ['HIGH', 'LOW', 'ABNORMAL', 'INSUFFICIENT']]

            for test in abnormal:
                flag = test.get('flag', '')
                card_class = 'danger-card' if flag in ['HIGH', 'ABNORMAL'] else 'warning-card'

                st.markdown(f"""
                <div class="info-card {card_class}" style="color: #2c3e50;">
                    <h4 style="color: #2c3e50;">{test.get('name', 'N/A')}</h4>
                    <p style="color: #2c3e50;"><strong>Result:</strong> {test.get('result', 'N/A')} {test.get('unit', '')} [{flag}]</p>
                    <p style="color: #2c3e50;"><strong>Reference:</strong> {test.get('reference_range', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("All test results are within normal range")

        st.markdown("---")

        # Report details
        st.markdown("### Report Information")

        report_info = st.session_state.lab_data.get('report_info', {})
        clinic_info = st.session_state.lab_data.get('clinic', {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Collection Date:**")
            st.write(report_info.get('collection_date', 'N/A'))

            st.markdown("**Report Date:**")
            st.write(report_info.get('report_date', 'N/A'))

        with col2:
            st.markdown("**Doctor:**")
            st.write(clinic_info.get('doctor', 'N/A'))

            st.markdown("**Report Type:**")
            st.write(report_info.get('report_type', 'N/A'))

    with tab3:
        # Patient info
        patient = st.session_state.lab_data.get('patient', {})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Patient", patient.get('name', 'N/A'))
        with col2:
            st.metric("Age", patient.get('age', 'N/A'))
        with col3:
            st.metric("Gender", patient.get('gender', 'N/A'))

        st.markdown("---")

        # Filter options
        col1, col2 = st.columns([1, 3])
        with col1:
            filter_option = st.selectbox(
                "Filter Results",
                ["All Tests", "Abnormal Only", "Normal Only"]
            )

        # Prepare test data
        tests = st.session_state.lab_data.get('tests', [])

        # Filter based on selection
        if filter_option == "Abnormal Only":
            tests = [t for t in tests if t.get('flag') in ['HIGH', 'LOW', 'ABNORMAL', 'INSUFFICIENT']]
        elif filter_option == "Normal Only":
            tests = [t for t in tests if t.get('flag') not in ['HIGH', 'LOW', 'ABNORMAL', 'INSUFFICIENT'] or t.get('flag') is None]

        # Display tests in a nice table
        if tests:
            # Group by category
            categories = {}
            for test in tests:
                cat = test.get('category', 'Other')
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(test)

            for category, cat_tests in categories.items():
                with st.expander(f"{category} ({len(cat_tests)} tests)", expanded=True):
                    # Create DataFrame
                    df_data = []
                    for test in cat_tests:
                        flag = test.get('flag', 'NORMAL')
                        if flag is None:
                            flag = 'NORMAL'

                        df_data.append({
                            'Test Name': test.get('name', 'N/A'),
                            'Result': f"{test.get('result', 'N/A')} {test.get('unit', '')}",
                            'Reference Range': test.get('reference_range', 'N/A'),
                            'Status': flag
                        })

                    df = pd.DataFrame(df_data)

                    # Style the dataframe
                    def highlight_status(val):
                        if val in ['HIGH', 'LOW', 'ABNORMAL', 'INSUFFICIENT']:
                            return 'background-color: #ffebee; color: #c62828; font-weight: bold'
                        elif val == 'NORMAL':
                            return 'background-color: #e8f5e9; color: #2e7d32; font-weight: bold'
                        else:
                            return ''

                    styled_df = df.style.map(highlight_status, subset=['Status'])
                    st.dataframe(styled_df, width='stretch', hide_index=True)
        else:
            st.info("No tests found matching the filter criteria.")

    with tab4:
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Quick question suggestions (only show when no chat history)
        if len(st.session_state.chat_history) == 0:
            st.markdown("**Suggested Questions:**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Why is my CRP high?", width='stretch', key="q1"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "Why is my CRP high?"
                    })
                    with st.spinner("Thinking..."):
                        result = run_async(
                            st.session_state.orchestrator.process_question_with_logging(
                                "Why is my CRP high?",
                                user_id=st.session_state.user_id,
                                session_id=st.session_state.session_id
                            )
                        )
                        if result["status"] == "success":
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": result["answer"]
                            })
                            st.rerun()

                if st.button("What foods reduce inflammation?", width='stretch', key="q2"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "What foods reduce inflammation?"
                    })
                    with st.spinner("Thinking..."):
                        result = run_async(
                            st.session_state.orchestrator.process_question_with_logging(
                                "What foods reduce inflammation?",
                                user_id=st.session_state.user_id,
                                session_id=st.session_state.session_id
                            )
                        )
                        if result["status"] == "success":
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": result["answer"]
                            })
                            st.rerun()

            with col2:
                if st.button("What is Vitamin D good for?", width='stretch', key="q3"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "What is Vitamin D good for?"
                    })
                    with st.spinner("Thinking..."):
                        result = run_async(
                            st.session_state.orchestrator.process_question_with_logging(
                                "What is Vitamin D good for?",
                                user_id=st.session_state.user_id,
                                session_id=st.session_state.session_id
                            )
                        )
                        if result["status"] == "success":
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": result["answer"]
                            })
                            st.rerun()

                if st.button("Should I be concerned?", width='stretch', key="q4"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "Should I be concerned about my results?"
                    })
                    with st.spinner("Thinking..."):
                        result = run_async(
                            st.session_state.orchestrator.process_question_with_logging(
                                "Should I be concerned about my results?",
                                user_id=st.session_state.user_id,
                                session_id=st.session_state.session_id
                            )
                        )
                        if result["status"] == "success":
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": result["answer"]
                            })
                            st.rerun()

        # Chat input at bottom
        user_question = st.chat_input("Ask a question about your report...")

        if user_question:
            # Add user message to history and show immediately
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })

            # Display the user message immediately
            with st.chat_message("user"):
                st.markdown(user_question)

            # Get answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = run_async(
                        st.session_state.orchestrator.process_question_with_logging(
                            user_question,
                            user_id=st.session_state.user_id,
                            session_id=st.session_state.session_id
                        )
                    )

                    if result["status"] == "success":
                        st.markdown(result["answer"])
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["answer"]
                        })
                        st.rerun()

# Footer
st.markdown("---")
st.caption("This is an AI assistant for informational purposes only. Always consult your healthcare provider for medical advice.")
