# Medical Report AI Orchestrator

An AI-powered medical lab report analysis system that extracts, interprets, and answers questions about your health reports using multi-agent architecture and Google Gemini AI.

![Health Report Assistant](utils/648c6d3f-de6a-4d7a-871e-11684521f749.jpg)

## Overview

The Medical Report AI Orchestrator is a production-ready web application that helps patients understand their medical lab reports. It uses a sophisticated multi-agent AI system to:

- **Extract** structured data from unstructured PDF lab reports
- **Interpret** medical results in patient-friendly language
- **Answer** health-related questions about your reports
- **Provide** dietary recommendations and health insights

## Features

- **Intelligent PDF Processing**: Automatically extracts test results, patient information, and metadata from lab reports
- **Medical Interpretation**: Translates complex medical data into easy-to-understand explanations
- **Interactive Q&A**: Ask questions about your results and get AI-powered answers
- **Session Management**: Maintains conversation history for context-aware responses
- **Real-time Metrics**: Tracks test results and highlights abnormal values
- **Privacy-Focused**: Processes all data in-memory without persistent storage

## Architecture

This application uses a multi-agent architecture powered by Google's Agent Development Kit (ADK):

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Web UI                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Health Report Orchestrator                   │
│        (Session & Memory Management)                    │
└─────┬──────────────┬────────────────┬───────────────────┘
      │              │                │
      ▼              ▼                ▼
┌──────────┐  ┌─────────────┐  ┌─────────────┐
│Extractor │  │ Interpreter │  │  Q&A Agent  │
│  Agent   │  │    Agent    │  │             │
└──────────┘  └─────────────┘  └─────────────┘
      │              │                │
      └──────────────┴────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Google Gemini 2.0   │
          │    Flash Lite        │
          └──────────────────────┘
```

### Agent Responsibilities

1. **Extractor Agent**: Processes PDF lab reports and extracts structured data (test names, results, reference ranges, flags)
2. **Interpreter Agent**: Analyzes extracted data and generates patient-friendly medical interpretations
3. **General Q&A Agent**: Answers health-related questions based on the lab report context

## Prerequisites

- Python 3.8 or higher
- Google API Key for Gemini (Get one from [Google AI Studio](https://aistudio.google.com/app/apikey))
- Docker (optional, for containerized deployment)

## Quick Start

### Option 1: Run Locally with Python

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/medical-report-ai-orchestrator.git
   cd medical-report-ai-orchestrator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file in the project root:
   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   GOOGLE_CLOUD_LOCATION=global
   GOOGLE_GENAI_USE_VERTEXAI=0
   LOG_LEVEL=INFO
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

   Or on Windows:
   ```bash
   run_app.bat
   ```

5. **Access the application**

   Open your browser and navigate to `http://localhost:8501`

### Option 2: Run with Docker

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/medical-report-ai-orchestrator.git
   cd medical-report-ai-orchestrator
   ```

2. **Create `.env` file**

   Create a `.env` file with your Google API key:
   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   GOOGLE_CLOUD_LOCATION=global
   GOOGLE_GENAI_USE_VERTEXAI=0
   LOG_LEVEL=INFO
   ```

3. **Build the Docker image**
   ```bash
   docker build -t medical-report-ai .
   ```

4. **Run the container**
   ```bash
   docker run -p 8501:8501 --env-file .env medical-report-ai
   ```

5. **Access the application**

   Open your browser and navigate to `http://localhost:8501`

### Option 3: Docker Compose (Recommended)

1. **Clone and setup**
   ```bash
   git clone https://github.com/yourusername/medical-report-ai-orchestrator.git
   cd medical-report-ai-orchestrator
   ```

2. **Create `.env` file** (same as above)

3. **Run with Docker Compose**
   ```bash
   docker-compose up
   ```

4. **Access the application** at `http://localhost:8501`

## Usage

1. **Upload a Lab Report**: Click on the sidebar and upload a PDF lab report
2. **Analyze**: Click the "Analyze Report" button to process the document
3. **Review Results**:
   - View medical interpretation in the "Medical Interpretation" tab
   - See summary statistics in the "Summary" tab
   - Browse detailed test results in the "Lab Results" tab
4. **Ask Questions**: Use the "Chat" tab to ask questions about your results

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | Yes | - | Your Google AI API key for Gemini access |
| `GOOGLE_CLOUD_LOCATION` | No | `global` | Google Cloud region for API requests |
| `GOOGLE_GENAI_USE_VERTEXAI` | No | `0` | Use Vertex AI (1) or AI Studio (0) |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Getting a Google API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and add it to your `.env` file

## Project Structure

```
medical-report-ai-orchestrator/
│
├── agents/                          # AI Agent implementations
│   ├── extractor_agent.py          # PDF data extraction agent
│   ├── interpreter_agent.py        # Medical interpretation agent
│   ├── general_qa_agent.py         # Q&A agent
│   ├── orchestrator.py             # Basic orchestrator
│   ├── orchestrator_with_logging.py # Production orchestrator
│   └── orchestrator_with_memory.py # Memory-enabled orchestrator
│
├── tools/                           # Utility tools
│   └── pdf_utils.py                # PDF processing utilities
│
├── utils/                           # Helper utilities
│   └── 648c6d3f-de6a-4d7a-871e-11684521f749.jpg  # App icon
│
├── lab_reports/                     # Sample/test lab reports (gitignored)
│
├── app.py                          # Streamlit web application
├── config.py                       # Configuration management
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker container definition
├── docker-compose.yml              # Docker Compose configuration
├── .dockerignore                   # Docker build exclusions
├── .gitignore                      # Git exclusions
└── README.md                       # This file
```

## Technologies Used

- **Frontend**: [Streamlit](https://streamlit.io/) - Interactive web interface
- **AI Framework**: [Google ADK](https://github.com/google/generative-ai-python) - Agent Development Kit
- **AI Models**: [Google Gemini 2.0 Flash Lite](https://ai.google.dev/) - Fast, efficient AI
- **PDF Processing**: PyMuPDF, pdfplumber - PDF parsing and data extraction
- **Session Management**: In-memory session and memory services
- **Deployment**: Docker, Streamlit Cloud compatible

## Security & Privacy

- **No Persistent Storage**: All data is processed in-memory
- **API Key Security**: Keys are loaded from environment variables, never hardcoded
- **Medical Disclaimers**: All AI responses include appropriate medical disclaimers
- **HIPAA Considerations**: For production use, ensure compliance with healthcare regulations

## Limitations & Disclaimers

**IMPORTANT MEDICAL DISCLAIMER**

This application is for informational and educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.

- Always consult qualified healthcare professionals for medical decisions
- This tool does not provide medical diagnoses or prescriptions
- AI interpretations may contain errors or inaccuracies
- Lab report analysis should be reviewed by licensed medical professionals

## Development

### Running Tests

```bash
pytest
```

### Local Development Setup

1. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set up `.env` file with your API key

4. Run the application
   ```bash
   streamlit run app.py
   ```

## Deployment

### Streamlit Cloud

1. Push your repository to GitHub
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Add your `GOOGLE_API_KEY` in the Secrets section
5. Deploy!

### Google Cloud Platform

Use the included Dockerfile to deploy on:
- Google Cloud Run
- Google Kubernetes Engine (GKE)
- Google Compute Engine

### AWS

Deploy using:
- AWS ECS (Elastic Container Service)
- AWS Fargate
- AWS EC2

## Troubleshooting

### Common Issues

**Issue**: `GOOGLE_API_KEY not found in environment`
- **Solution**: Make sure you've created a `.env` file with your API key

**Issue**: PDF processing fails
- **Solution**: Ensure the PDF is not password-protected or corrupted

**Issue**: Docker container won't start
- **Solution**: Check that port 8501 is not already in use

**Issue**: Out of memory errors
- **Solution**: Increase Docker memory allocation or reduce PDF file size

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini AI for powering the intelligent agents
- Streamlit for the excellent web framework
- The open-source community for various PDF processing libraries

## Support

For questions, issues, or feature requests, please open an issue on GitHub.

---

**Built with ❤️ using Google Gemini AI and Streamlit**
