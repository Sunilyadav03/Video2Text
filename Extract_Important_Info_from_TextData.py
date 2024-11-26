# -- In this file we extract "customer_name, borrow_amount, repay_date, pan_number, aadhar_number, lender_name, credit_date" from financial text data.
import pdfplumber
import re
from transformers import pipeline
from datetime import datetime

# Initialize the question-answering pipeline with GPU support if available
nlp_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=0)

# Define specific questions for the question-answering model
qa_fields = {
    "customer_name": "What is the customer's full name?",
    "borrow_amount": "How much has the customer borrowed?",
    "repay_date": "What is the customer's repayment date?",
    "pan_number": "What is the customer's PAN number?",
    "aadhar_number": "What is the customer's Aadhar number?",
    "lender_name": "From where is the customer taking the loan?",
    "credit_date": "What is the credit date, borrow date, or disbursal date?"
}

# Use question-answering model for fields that require more context
extracted_data = {}
for key, question in qa_fields.items():
    result1 = nlp_pipeline({"question": question, "context": result})
    answer = result1["answer"]

    # Process specific fields for formatting
    if key == "borrow_amount":
        # Remove any commas in the amount
        answer = answer.replace(",", "")
    elif key in ["repay_date", "credit_date"]:
        # Standardize various date formats to DD/MM/YYYY
        # Match patterns like "3rd December 2024" or "4th November 2024"
        date_match = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})", answer)
        if date_match:
            day, month_str, year = date_match.groups()
            month = datetime.strptime(month_str[:3], "%b").month  # Convert month to numeric
            answer = f"{int(day)}/{month}/{year}"  # Format as DD/MM/YYYY
        else:
            # For formats like 31-10-2023 or 31/10/2023
            date_match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", answer)
            if date_match:
                day, month, year = date_match.groups()
                answer = f"{int(day)}/{int(month)}/{year}"  # Format as DD/MM/YYYY
    elif key in ["pan_number", "aadhar_number"]:
        # Ensure PAN/Aadhar is kept in the correct format
        answer = answer.replace(" ", "")  # Remove spaces if needed

    extracted_data[key] = answer

# Print final extracted information
print("Extracted Bank Statement Details:")
print(f"Customer Name: {extracted_data.get('customer_name')}")
print(f"Borrow Amount: {extracted_data.get('borrow_amount')}")
print(f"Repay Date: {extracted_data.get('repay_date')}")
print(f"Lender Name: {extracted_data.get('lender_name')}")
print(f"Credit Date: {extracted_data.get('credit_date')}")
