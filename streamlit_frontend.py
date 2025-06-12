
import streamlit as st
import requests
import uuid
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Set page configuration first
st.set_page_config(page_title="NIST AI RMF Compliance Checker", page_icon="✨")

# Title
st.title("NIST AI RMF Compliance Checker")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm here to help you check compliance with the NIST AI RMF. Would you like to start?"
        }
    ]
if "results" not in st.session_state:
    st.session_state.results = []
if "report" not in st.session_state:
    st.session_state.report = None

# Function to convert report to PDF
def create_pdf(report_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    style = styles["BodyText"]
    
    # Split report into paragraphs and create PDF elements
    elements = []
    for line in report_text.split("\n"):
        if line.strip():
            elements.append(Paragraph(line, style))
            elements.append(Spacer(1, 12))
    
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# Chat container
chat_container = st.container()

# Display chat history
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="✨" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])

# Chat input at the bottom
user_input = st.chat_input("Type your response...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Rerender chat history with new user message
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="✨" if msg["role"] == "assistant" else None):
                st.markdown(msg["content"])
    
    # Make API call to the FastAPI backend
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"text": user_input, "session_id": st.session_state.session_id}
        )
        response.raise_for_status()
        response_data = response.json()
        
        # Get assistant's response
        assistant_response = response_data["content"]
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        # Check if the response contains a checklist
        if "Here’s a summary of your answers:" in assistant_response:
            checklist_start = assistant_response.find("Here’s a summary of your answers:") + len("Here’s a summary of your answers:\n")
            checklist_end = assistant_response.find("\n\nWould you like to generate a summary report")
            if checklist_end != -1:
                checklist_text = assistant_response[checklist_start:checklist_end]
            else:
                checklist_text = assistant_response[checklist_start:]
            
            lines = checklist_text.strip().split("\n")
            if len(lines) > 2:
                headers = [h.strip() for h in lines[1].split("|")[1:-1]]
                data = []
                for line in lines[3:-1]:
                    row = [cell.strip() for cell in line.split("|")[1:-1]]
                    data.append(row)
                df = pd.DataFrame(data, columns=headers)
                for _, row in df.iterrows():
                    st.session_state.results.append({
                        "Question Number": row["#"],
                        "Question": row["Question"],
                        "Status": row["Status"],
                        "Comment": row["Comment"]
                    })
        
        # Check if the response contains a report
        if "### Summary Report" in assistant_response:
            report_start = assistant_response.find("### Summary Report\n\n") + len("### Summary Report\n\n")
            report_end = assistant_response.find("\n\nYou can download this report")
            if report_end != -1:
                report_text = assistant_response[report_start:report_end]
            else:
                report_text = assistant_response[report_start:]
            st.session_state.report = report_text
        
        # Rerender chat with the assistant's response
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar="✨" if msg["role"] == "assistant" else None):
                    st.markdown(msg["content"])
                    
    except requests.RequestException as e:
        st.error(f"Error connecting to the backend: {e}")
    
    # Rerun to update the UI
    st.rerun()

# Display checklist
if st.button("Show Checklist"):
    if st.session_state.results:
        st.subheader("Compliance Checklist")
        df = pd.DataFrame(st.session_state.results)
        st.dataframe(df)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Checklist as CSV",
            data=csv,
            file_name=f"nist_rmf_checklist_{st.session_state.session_id}.csv",
            mime="text/csv"
        )
    else:
        st.info("No results to display yet. Please complete a policy creation session first.")

# Display report and download button
if st.session_state.report:
    st.subheader("Governance Report")
    st.markdown(st.session_state.report)
    pdf_data = create_pdf(st.session_state.report)
    st.download_button(
        label="Download Report as PDF",
        data=pdf_data,
        file_name=f"nist_rmf_report_{st.session_state.session_id}.pdf",
        mime="application/pdf"
    )

# Clear chat history
if st.button("Clear Chat"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm here to help you check compliance with the NIST AI RMF. Would you like to start?"
        }
    ]
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.results = []
    st.session_state.report = None
    st.rerun()