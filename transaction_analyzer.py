import streamlit as st
import pandas as pd
import plotly.express as px
import uuid
from datetime import datetime
import pdfplumber
from PyPDF2 import PdfReader
import json
import re
from openai import OpenAI

# Global OpenAI client instance
client = None

def get_transactions_df():
    """Safely get the transactions DataFrame from session state"""
    if 'transactions' not in st.session_state:
        st.session_state['transactions'] = pd.DataFrame(columns=[
            'id', 'date', 'amount', 'category', 'description', 'merchant', 'transaction_type', 'source'
        ])
    return st.session_state['transactions']

def has_transactions():
    """Safely check if there are any transactions"""
    try:
        df = get_transactions_df()
        return len(df) > 0 and not df.empty
    except Exception as e:
        st.error(f"Error checking transactions: {e}")
        return False

def clear_all_transactions():
    """Safely clear all transactions"""
    st.session_state['transactions'] = pd.DataFrame(columns=[
        'id', 'date', 'amount', 'category', 'description', 'merchant', 'transaction_type', 'source'
    ])

def add_transactions(new_df):
    """Safely add new transactions to the session state"""
    current_df = get_transactions_df()
    if current_df.empty:
        st.session_state['transactions'] = new_df
    else:
        st.session_state['transactions'] = pd.concat([current_df, new_df], ignore_index=True)

def delete_transaction(transaction_id):
    """Safely delete a specific transaction"""
    current_df = get_transactions_df()
    st.session_state['transactions'] = current_df[current_df['id'] != transaction_id]

def get_openai_client(api_key: str):
    global client
    if api_key:
        client = OpenAI(api_key=api_key)
        return True
    else:
        client = None
        return False

def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
            return text
    except Exception:
        try:
            pdf_file.seek(0)
            reader = PdfReader(pdf_file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            return text
        except Exception as e:
            st.error(f"Failed to extract text from PDF: {e}")
            return ""

def fix_incomplete_json(json_str):
    """Attempt to fix incomplete JSON strings"""
    json_str = json_str.strip()
    
    # Remove markdown formatting if present
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()
    
    # Try to fix common incomplete JSON issues
    if not json_str.endswith(']') and not json_str.endswith('}'):
        # If it looks like an incomplete array
        if json_str.startswith('['):
            # Count open braces to try to close them
            open_braces = json_str.count('{') - json_str.count('}')
            json_str += '}' * open_braces
            if not json_str.endswith(']'):
                json_str += ']'
        # If it looks like an incomplete object
        elif json_str.startswith('{'):
            open_braces = json_str.count('{') - json_str.count('}')
            json_str += '}' * open_braces
    
    return json_str

def extract_date_patterns_from_text(text):
    """Extract potential date patterns from text to help OpenAI"""
    import re
    
    # Common Australian date patterns
    patterns = [
        r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\b',  # DD/MM/YYYY or DD-MM-YYYY
        r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2})\b',   # DD/MM/YY or DD-MM-YY
        r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b',  # DD MMM YYYY
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})\s+(\d{4})\b',  # MMM DD YYYY
    ]
    
    found_dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found_dates.extend(matches)
    
    return found_dates[:10]  # Return first 10 date patterns found

def extract_transactions_from_text(text):
    # Show the user what we're working with
    st.write(f"📄 **PDF Text Length:** {len(text)} characters")
    
    # First, let's chunk the text if it's too long (OpenAI has token limits)
    max_chars = 12000  # Leave room for the prompt
    text_chunks = []
    
    if len(text) > max_chars:
        st.warning(f"⚠️ PDF text is long ({len(text)} chars). Splitting into chunks for better processing.")
        
        # Split by pages or sections
        pages = text.split('\n\n')  # Assume double newlines separate sections
        current_chunk = ""
        
        for page in pages:
            if len(current_chunk + page) < max_chars:
                current_chunk += page + "\n\n"
            else:
                if current_chunk:
                    text_chunks.append(current_chunk)
                current_chunk = page + "\n\n"
        
        if current_chunk:
            text_chunks.append(current_chunk)
    else:
        text_chunks = [text]
    
    st.write(f"📝 **Processing {len(text_chunks)} text chunks**")
    
    all_transactions = []
    
    for i, chunk in enumerate(text_chunks):
        st.write(f"🔄 **Processing chunk {i+1}/{len(text_chunks)}** ({len(chunk)} characters)")
        
        # Show chunk preview
        with st.expander(f"📖 Preview Chunk {i+1}", expanded=False):
            st.code(chunk[:500] + "..." if len(chunk) > 500 else chunk)
        
        # Extract date context from this chunk
        date_patterns = extract_date_patterns_from_text(chunk)
        
        # Enhanced prompt for bank statements
        prompt = f"""
You are an expert at extracting transactions from Australian bank statements. 

CONTEXT for this chunk:
- This is chunk {i+1} of {len(text_chunks)} from a bank statement
- Date patterns found: {date_patterns if date_patterns else "None detected"}
- You must extract ALL transactions from this text chunk

AUSTRALIAN BANK STATEMENT RULES:
1. Dates are usually in DD/MM/YYYY or DD/MM/YY format
2. Transactions are listed chronologically (oldest to newest or newest to oldest)
3. Each line with a date and amount is likely a transaction
4. Look for: Date, Description, Debit/Credit amounts
5. Common sections: "Transaction Details", "Account Summary", etc.
6. Ignore headers, footers, and summary lines

DATE RULES:
- If you see "01/01" or "1/1", it's January 1st
- If you see "15/08" or "15/8", it's August 15th  
- If year is missing, assume 2024
- Each transaction should have its ACTUAL date from the statement

AMOUNT RULES:
- Debits (money out) should be negative numbers
- Credits (money in) should be positive numbers
- Look for CR (credit) or DR (debit) indicators

Extract ALL transactions from this bank statement chunk. Return a JSON array:

[
  {{
    "date": "YYYY-MM-DD", 
    "amount": number,
    "description": "actual description from statement",
    "merchant": "merchant name if identifiable",
    "category": "Food|Shopping|Transport|Bills|Income|Other",
    "transaction_type": "credit|debit"
  }}
]

Bank statement text chunk:
```
{chunk}
```

Return ONLY the JSON array. Extract every single transaction you can find.
"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = response.choices[0].message.content.strip()
            
            st.write(f"🤖 **OpenAI Response Length:** {len(content)} characters")
            
            # Show raw response in expander
            with st.expander(f"🔍 Raw OpenAI Response (Chunk {i+1})", expanded=False):
                st.code(content[:1000] + "..." if len(content) > 1000 else content)
            
            # Fix the content if it's wrapped in markdown or incomplete
            content = fix_incomplete_json(content)
            
            try:
                chunk_transactions = json.loads(content)
                if not isinstance(chunk_transactions, list):
                    chunk_transactions = [chunk_transactions]
                
                st.success(f"✅ **Chunk {i+1}: Extracted {len(chunk_transactions)} transactions**")
                
                # Show sample from this chunk
                if chunk_transactions:
                    sample_tx = chunk_transactions[0]
                    st.write(f"   📋 Sample: {sample_tx.get('date', 'No date')} - {sample_tx.get('description', 'No desc')[:30]}... - ${sample_tx.get('amount', 0)}")
                
                all_transactions.extend(chunk_transactions)
                
            except json.JSONDecodeError as e:
                st.error(f"❌ **Chunk {i+1}: JSON parsing failed:** {e}")
                st.code(content[:500])
                continue
                
        except Exception as e:
            st.error(f"❌ **Chunk {i+1}: OpenAI error:** {e}")
            continue
    
    st.write(f"🎯 **Total transactions extracted from all chunks:** {len(all_transactions)}")
    
    # Validate and clean all transactions
    cleaned_transactions = []
    for i, transaction in enumerate(all_transactions):
        # Ensure all required fields exist
        cleaned_tx = {
            'amount': transaction.get('amount', 0.0),
            'category': transaction.get('category', 'Other'),
            'description': transaction.get('description', f'Transaction {i+1}'),
            'merchant': transaction.get('merchant', transaction.get('description', 'Unknown')),
            'transaction_type': transaction.get('transaction_type', 'debit' if transaction.get('amount', 0) < 0 else 'credit'),
            'date': transaction.get('date', '2024-01-01')  # Default fallback
        }
        cleaned_transactions.append(cleaned_tx)
    
    return cleaned_transactions


def main():
    # Initialize session state FIRST before anything else
    _ = get_transactions_df()  # This will initialize if needed
    
    st.title("💰 Smart Transaction Tracker with PDF & AI")

    # Sidebar for OpenAI API and AI Chat
    with st.sidebar:
        st.header("🔧 Settings")
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key to enable PDF extraction and AI features")
        openai_ready = get_openai_client(api_key)
        
        if openai_ready:
            st.success("✅ OpenAI connected!")
        else:
            st.warning("⚠️ Enter API key to enable AI features")

        st.markdown("---")
        
        # AI Chat Section
        if openai_ready and has_transactions():
            st.header("🤖 Ask AI About Your Finances")
            
            # Quick question buttons
            st.subheader("Quick Questions:")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💸 Biggest Expenses", key="biggest_exp"):
                    st.session_state['ai_question'] = "What are my biggest expense categories and how much did I spend on each?"
                if st.button("📈 Spending Trends", key="trends"):
                    st.session_state['ai_question'] = "Analyze my spending patterns and trends"
            with col2:
                if st.button("💰 Income Summary", key="income"):
                    st.session_state['ai_question'] = "Summarize my income sources and amounts"
                if st.button("💡 Savings Tips", key="tips"):
                    st.session_state['ai_question'] = "Give me personalized tips to save money based on my spending"
            
            # Custom question input
            question = st.text_area(
                "Or ask a custom question:", 
                value=st.session_state.get('ai_question', ''),
                placeholder="e.g., How much did I spend on food last month?",
                key="question_input"
            )

            if st.button("🚀 Get AI Answer", key="ai_answer_btn") and question:
                with st.spinner("AI is analyzing your data..."):
                    df = get_transactions_df()
                    
                    # Create a comprehensive summary
                    summary = f"""
Transaction Summary:
- Total transactions: {len(df)}
- Total income: ${df[df['amount'] > 0]['amount'].sum():,.2f}
- Total expenses: ${abs(df[df['amount'] < 0]['amount'].sum()):,.2f}
- Net amount: ${df['amount'].sum():,.2f}
- Categories: {', '.join(df['category'].unique())}
- Date range: {df['date'].min()} to {df['date'].max()}

Sample transactions:
{df.head(10).to_dict(orient='records')}
"""
                    
                    prompt = f"""
You are a helpful financial assistant. Here is the user's transaction data:

{summary}

Based on this data, answer the user's question clearly and provide actionable insights. Be specific with numbers and dates when possible.

User's question: {question}
"""
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                        )
                        answer = response.choices[0].message.content.strip()
                        
                        st.markdown("### 🤖 AI Answer:")
                        st.markdown(answer)
                        
                        # Clear the question after getting answer
                        if 'ai_question' in st.session_state:
                            del st.session_state['ai_question']
                            
                    except Exception as e:
                        st.error(f"Error generating AI answer: {e}")
        
        elif has_transactions():
            st.info("💡 Enter your OpenAI API key above to unlock AI-powered financial insights!")

    # Debug info (remove this after fixing)
    with st.expander("🔍 Debug Info", expanded=False):
        st.write(f"**Session State Status:**")
        st.write(f"- Transactions DataFrame shape: {st.session_state['transactions'].shape}")
        st.write(f"- Is DataFrame empty: {st.session_state['transactions'].empty}")
        st.write(f"- DataFrame columns: {list(st.session_state['transactions'].columns)}")
        if has_transactions():
            st.write("**Sample data:**")
            st.dataframe(st.session_state['transactions'].head(3))
        
        # Test button to add sample data
        if st.button("🧪 Add Test Transactions"):
            test_data = [
                {
                    'id': str(uuid.uuid4()),
                    'date': pd.to_datetime('2024-01-15'),
                    'amount': -25.50,
                    'category': 'Food',
                    'description': 'Coffee shop',
                    'merchant': 'Starbucks',
                    'transaction_type': 'debit',
                    'source': 'test_data'
                },
                {
                    'id': str(uuid.uuid4()),
                    'date': pd.to_datetime('2024-01-16'),
                    'amount': 2500.00,
                    'category': 'Income',
                    'description': 'Salary',
                    'merchant': 'Company Inc',
                    'transaction_type': 'credit',
                    'source': 'test_data'
                }
            ]
            add_transactions(pd.DataFrame(test_data))
            st.success("Test data added!")
            st.rerun()

    # Date Correction Tool
    if has_transactions():
        with st.expander("🔧 Manual Date Correction Tool", expanded=False):
            df = get_transactions_df()
            
            st.write("**Fix incorrect dates from PDF extraction:**")
            
            # Show current date distribution
            if 'date' in df.columns:
                date_summary = df['date'].dropna()
                if len(date_summary) > 0:
                    month_counts = date_summary.dt.to_period('M').value_counts().sort_index()
                    st.write("📊 Current date distribution:")
                    for month, count in month_counts.items():
                        st.write(f"  • {month}: {count} transactions")
                else:
                    st.warning("No valid dates found in transactions")
            
            # Quick date fixes
            st.write("**Quick Date Corrections:**")
            col1, col2 = st.columns(2)
            
            with col1:
                # Fix all transactions to a specific month
                fix_month = st.selectbox("Set all transactions to month:", [
                    "Don't change", "2024-01 (January)", "2024-02 (February)", 
                    "2024-03 (March)", "2024-04 (April)", "2024-05 (May)", 
                    "2024-06 (June)", "2024-07 (July)", "2024-08 (August)"
                ])
                
                if st.button("🔄 Apply Month Fix") and fix_month != "Don't change":
                    month_str = fix_month.split(" ")[0]  # Extract YYYY-MM
                    current_df = get_transactions_df()
                    # Set all transactions to the 15th of the selected month
                    current_df['date'] = pd.to_datetime(f"{month_str}-15")
                    st.session_state['transactions'] = current_df
                    st.success(f"All transactions set to {fix_month}")
                    st.rerun()
            
            with col2:
                # Spread transactions across multiple months
                if st.button("📅 Spread Across Jan-Aug 2024"):
                    current_df = get_transactions_df()
                    num_transactions = len(current_df)
                    
                    # Create date range from Jan to Aug 2024
                    start_date = pd.to_datetime("2024-01-01")
                    end_date = pd.to_datetime("2024-08-31")
                    date_range = pd.date_range(start_date, end_date, periods=num_transactions)
                    
                    current_df['date'] = date_range
                    st.session_state['transactions'] = current_df
                    st.success("Transactions spread across January to August 2024")
                    st.rerun()
            
            # Advanced: Manual date editing for specific transactions
            st.write("**Manual Transaction Date Editing:**")
            if st.checkbox("Enable manual date editing"):
                df = get_transactions_df()
                if not df.empty:
                    # Show transactions with editable dates
                    st.write("Click on a transaction to edit its date:")
                    for idx, row in df.head(10).iterrows():  # Show first 10 for editing
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.write(f"**{row['description']}** - ${row['amount']:.2f}")
                        with col2:
                            current_date = row['date']
                            if pd.isna(current_date):
                                current_date = datetime.now().date()
                            else:
                                current_date = current_date.date()
                            st.write(f"Current: {current_date}")
                        with col3:
                            new_date = st.date_input(f"New date", value=current_date, key=f"date_edit_{idx}")
                            if st.button("Update", key=f"update_{idx}"):
                                current_df = get_transactions_df()
                                current_df.loc[idx, 'date'] = pd.to_datetime(new_date)
                                st.session_state['transactions'] = current_df
                                st.success(f"Updated date for transaction {idx}")
                                st.rerun()

    # SIMPLE SOLUTION - Manual CSV Upload
    st.header("🚀 QUICK FIX: Manual Transaction Entry")
    
    # Add a big obvious section for manual solutions
    with st.container():
        st.markdown("""
        <div style="background: linear-gradient(90deg, #ff6b6b, #4ecdc4); padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h2 style="color: white; margin: 0;">😤 PDF Not Working? Try These INSTANT Solutions:</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Year selection for sample data
        st.subheader("📅 First, what year are your transactions from?")
        year_choice = st.radio("Select the year of your bank statement:", 
                              ["2025 (Current Year)", "2024 (Last Year)"], 
                              horizontal=True)
        
        selected_year = 2025 if "2025" in year_choice else 2024
        st.success(f"✅ Using year: **{selected_year}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Option 1: Create Sample Data")
            if st.button(f"🎯 Generate Test Data (Jan-Aug {selected_year})", type="primary", key="quick_fix"):
                # Create realistic sample transactions from Jan to Aug
                sample_transactions = []
                
                # Generate transactions for each month
                months = [f'{selected_year}-01', f'{selected_year}-02', f'{selected_year}-03', f'{selected_year}-04', 
                         f'{selected_year}-05', f'{selected_year}-06', f'{selected_year}-07', f'{selected_year}-08']
                descriptions = [
                    'Woolworths Supermarket', 'Coles Grocery', 'Coffee Club', 'Shell Petrol', 
                    'Salary Deposit', 'McDonald\'s', 'Uber Ride', 'Netflix Subscription',
                    'Electricity Bill', 'Phone Bill', 'ATM Withdrawal', 'Online Shopping',
                    'Restaurant Dinner', 'Gym Membership', 'Insurance Payment', 'Rent Payment'
                ]
                
                for month in months:
                    # Add 8-12 transactions per month
                    import random
                    num_txns = random.randint(8, 12)
                    
                    for i in range(num_txns):
                        day = random.randint(1, 28)
                        desc = random.choice(descriptions)
                        
                        # Mix of income and expenses
                        if 'Salary' in desc or 'Deposit' in desc:
                            amount = random.randint(2000, 4000)  # Income
                            tx_type = 'credit'
                        else:
                            amount = -random.randint(10, 300)  # Expenses
                            tx_type = 'debit'
                        
                        transaction = {
                            'id': str(uuid.uuid4()),
                            'date': pd.to_datetime(f"{month}-{day:02d}"),
                            'amount': amount,
                            'category': 'Food' if any(x in desc for x in ['Woolworths', 'Coles', 'McDonald', 'Coffee', 'Restaurant']) 
                                       else 'Transport' if any(x in desc for x in ['Shell', 'Uber']) 
                                       else 'Bills' if any(x in desc for x in ['Bill', 'Netflix', 'Insurance', 'Rent'])
                                       else 'Income' if amount > 0 else 'Other',
                            'description': desc,
                            'merchant': desc,
                            'transaction_type': tx_type,
                            'source': f'sample_data_{selected_year}'
                        }
                        sample_transactions.append(transaction)
                
                # Add all sample transactions
                clear_all_transactions()  # Clear existing data
                add_transactions(pd.DataFrame(sample_transactions))
                
                st.success(f"🎉 Created {len(sample_transactions)} sample transactions from January to August {selected_year}!")
                st.balloons()
                st.rerun()
        
        with col2:
            st.subheader("📝 Option 2: Manual CSV Upload")
            st.write("Create a CSV file with columns: date, amount, description, category")
            
            # Provide downloadable template with correct year
            template_data = {
                'date': [f'{selected_year}-01-15', f'{selected_year}-02-20', f'{selected_year}-03-10', 
                        f'{selected_year}-04-05', f'{selected_year}-05-12', f'{selected_year}-06-18', 
                        f'{selected_year}-07-22', f'{selected_year}-08-08'],
                'amount': [-45.67, 2500.00, -89.50, -23.40, -156.78, 3000.00, -67.89, -234.56],
                'description': ['Woolworths', 'Salary', 'Coles', 'Coffee', 'Petrol', 'Bonus', 'Dinner', 'Shopping'],
                'category': ['Food', 'Income', 'Food', 'Food', 'Transport', 'Income', 'Food', 'Shopping']
            }
            template_df = pd.DataFrame(template_data)
            
            st.write(f"📋 **Template Example ({selected_year}):**")
            st.dataframe(template_df)
            
            # Download template
            csv = template_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download CSV Template ({selected_year})",
                data=csv,
                file_name=f"transaction_template_{selected_year}.csv",
                mime="text/csv"
            )
    
    # Year correction tool for existing data
    if has_transactions():
        st.subheader("🔧 Fix Year Issues in Existing Data")
        current_df = get_transactions_df()
        
        if 'date' in current_df.columns:
            # Show current year distribution
            year_counts = current_df['date'].dropna().dt.year.value_counts().sort_index()
            if len(year_counts) > 0:
                st.write("📊 **Current years in your data:**")
                for year, count in year_counts.items():
                    st.write(f"   • **{year}:** {count} transactions")
                
                # Year correction
                if len(year_counts) > 1 or (2024 in year_counts.index and selected_year == 2025):
                    st.warning("⚠️ **Mixed years detected!** Let's fix this:")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🔄 Convert ALL dates to {selected_year}", type="primary"):
                            current_df = get_transactions_df()
                            # Change only the year, keep month and day
                            current_df['date'] = current_df['date'].apply(
                                lambda x: x.replace(year=selected_year) if pd.notna(x) else x
                            )
                            st.session_state['transactions'] = current_df
                            st.success(f"✅ All dates converted to {selected_year}!")
                            st.rerun()
                    
                    with col2:
                        if st.button("🗑️ Clear All & Start Fresh"):
                            clear_all_transactions()
                            st.success("🧹 All data cleared!")
                            st.rerun()
    
    # Simplified PDF section
    st.header("📁 Add Transaction Data")

    input_method = st.radio(
        "Choose input method:",
        ["Upload PDF", "Upload CSV", "Manual Entry"],
        horizontal=True
    )

    if input_method == "Upload PDF":
        st.subheader("📄 Upload Bank Statement PDFs")
        uploaded_files = st.file_uploader(
            "Upload one or multiple PDF files", 
            accept_multiple_files=True, 
            type=["pdf"],
            help="Upload bank statements, credit card statements, or other financial PDFs"
        )
        
        if uploaded_files and openai_ready:
            if st.button("🔄 Extract Transactions from PDFs", type="primary"):
                progress_bar = st.progress(0)
                all_transactions = []
                
                # Show first part of PDF text for debugging
                if st.checkbox("🔍 Show PDF text preview (for debugging)", key="show_pdf_text"):
                    pdf_file = uploaded_files[0]  # Show first file
                    preview_text = extract_text_from_pdf(pdf_file)
                    if preview_text:
                        st.write("📄 PDF Text Preview (first 1000 characters):")
                        st.code(preview_text[:1000])
                        
                        # Show date patterns found
                        date_patterns = extract_date_patterns_from_text(preview_text)
                        if date_patterns:
                            st.write(f"🗓️ Date patterns detected: {date_patterns}")
                        else:
                            st.warning("⚠️ No date patterns detected in PDF text")
                
                for i, pdf_file in enumerate(uploaded_files):
                    st.info(f"Processing {pdf_file.name}...")
                    text = extract_text_from_pdf(pdf_file)
                    
                    if text:
                        st.write(f"📄 Extracted text length: {len(text)} characters")
                        txns = extract_transactions_from_text(text)
                        
                        if txns:
                            # Add metadata to each transaction
                            for t in txns:
                                t['source'] = pdf_file.name
                                t['id'] = str(uuid.uuid4())
                            all_transactions.extend(txns)
                            st.success(f"✅ Extracted {len(txns)} transactions from {pdf_file.name}")
                        else:
                            st.warning(f"⚠️ No transactions extracted from {pdf_file.name}")
                    else:
                        st.error(f"❌ No text extracted from {pdf_file.name}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.write(f"🔍 Total transactions collected: {len(all_transactions)}")
                
                if all_transactions:
                    # Create new DataFrame from transactions
                    new_df = pd.DataFrame(all_transactions)
                    st.write(f"📊 New DataFrame shape: {new_df.shape}")
                    st.write("📋 Sample transactions:")
                    st.dataframe(new_df.head())
                    
                    # Convert date column
                    if 'date' in new_df.columns:
                        new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
                    
                    # Check current session state
                    st.write(f"📈 Current transactions in session: {len(st.session_state['transactions'])}")
                    
                    # Add to session state
                    if st.session_state['transactions'].empty:
                        st.session_state['transactions'] = new_df
                    else:
                        st.session_state['transactions'] = pd.concat([st.session_state['transactions'], new_df], ignore_index=True)
                    
                    st.write(f"📊 Total transactions after adding: {len(st.session_state['transactions'])}")
                    st.balloons()
                    st.success(f"🎉 Successfully added {len(all_transactions)} transactions!")
                    
                    # Force page refresh to show the data
                    st.rerun()
                else:
                    st.warning("No transactions were extracted from the uploaded PDFs.")

        elif uploaded_files and not openai_ready:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to extract transactions from PDFs.")

    elif input_method == "📊 Upload CSV":
        st.subheader("📊 Upload CSV File")
        st.info("💡 Use the template download above to create your CSV file")
        
        uploaded_csv = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_csv:
            try:
                new_df = pd.read_csv(uploaded_csv)
                st.write("✅ **CSV Preview:**")
                st.dataframe(new_df.head())
                
                if st.button("➕ Add These Transactions", type="primary"):
                    # Add required fields
                    if 'id' not in new_df.columns:
                        new_df['id'] = [str(uuid.uuid4()) for _ in range(len(new_df))]
                    if 'merchant' not in new_df.columns:
                        new_df['merchant'] = new_df.get('description', 'Unknown')
                    if 'transaction_type' not in new_df.columns:
                        new_df['transaction_type'] = new_df['amount'].apply(lambda x: 'credit' if x > 0 else 'debit')
                    if 'source' not in new_df.columns:
                        new_df['source'] = uploaded_csv.name
                    
                    # Convert date
                    new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
                    
                    add_transactions(new_df)
                    st.success(f"🎉 Added {len(new_df)} transactions!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"CSV Error: {e}")

    elif input_method == "✍️ Manual Entry":
        st.subheader("✍️ Add Single Transaction")
        
        with st.form("add_transaction"):
            col1, col2 = st.columns(2)
            with col1:
                date = st.date_input("Date", value=datetime.now().date())
                amount = st.number_input("Amount ($)", step=0.01, help="Negative for expenses, positive for income")
            with col2:
                description = st.text_input("Description", placeholder="e.g., Woolworths, Salary, Coffee")
                category = st.selectbox("Category", ["Food", "Transport", "Bills", "Income", "Shopping", "Other"])
            
            if st.form_submit_button("➕ Add Transaction", type="primary"):
                new_tx = {
                    'id': str(uuid.uuid4()),
                    'date': pd.to_datetime(date),
                    'amount': amount,
                    'category': category,
                    'description': description,
                    'merchant': description,
                    'transaction_type': "credit" if amount > 0 else "debit",
                    'source': 'manual_entry'
                }
                add_transactions(pd.DataFrame([new_tx]))
                st.success("✅ Transaction added!")
                st.rerun()

    # Display transactions
    st.markdown("---")
    st.header("📋 Transaction Data")
    
    if has_transactions():
        df = st.session_state['transactions']
        
        # View options
        view_type = st.radio("Choose view:", ["Table View", "Card View", "Monthly Summary"], horizontal=True)
        
        # Date-based filtering
        st.subheader("📅 Filter by Date")
        col1, col2, col3 = st.columns(3)
        
        # Get available date range
        df_with_dates = df.dropna(subset=['date'])
        if not df_with_dates.empty:
            min_date = df_with_dates['date'].min()
            max_date = df_with_dates['date'].max()
            
            with col1:
                filter_type = st.selectbox("Date Filter Type", [
                    "All Time", 
                    "Specific Month", 
                    "Date Range", 
                    "Year"
                ])
            
            with col2:
                if filter_type == "Specific Month":
                    # Extract unique year-month combinations
                    available_months = df_with_dates['date'].dt.to_period('M').unique()
                    available_months = sorted(available_months, reverse=True)
                    month_options = [str(month) for month in available_months]
                    
                    if month_options:
                        selected_month = st.selectbox("Select Month", month_options)
                    else:
                        st.warning("No months with valid dates found")
                        selected_month = None
                        
                elif filter_type == "Year":
                    available_years = df_with_dates['date'].dt.year.unique()
                    available_years = sorted(available_years, reverse=True)
                    if len(available_years) > 0:
                        selected_year = st.selectbox("Select Year", available_years)
                    else:
                        st.warning("No years with valid dates found")
                        selected_year = None
            
            with col3:
                if filter_type == "Date Range":
                    start_date = st.date_input("Start Date", value=min_date.date())
                    end_date = st.date_input("End Date", value=max_date.date())
        
        # Quick search for dates and descriptions
        st.subheader("🔍 Search Transactions")
        search_query = st.text_input("Search by description, merchant, or date (e.g., '01' for January, 'coffee', 'woolworths'):", 
                                   placeholder="Try: 01, Feb, coffee, shopping...")
        
        if search_query:
            search_lower = search_query.lower().strip()
            
            # Search in multiple fields
            search_mask = (
                filtered_df['description'].str.lower().str.contains(search_lower, na=False) |
                filtered_df['merchant'].str.lower().str.contains(search_lower, na=False) |
                filtered_df['category'].str.lower().str.contains(search_lower, na=False)
            )
            
            # Special handling for date searches
            if search_query.isdigit() and len(search_query) <= 2:
                # User might be searching for month number (01, 02, etc.)
                month_num = int(search_query)
                if 1 <= month_num <= 12:
                    date_mask = filtered_df['date'].dt.month == month_num
                    search_mask = search_mask | date_mask
                    st.info(f"💡 Searching for month {month_num} and text containing '{search_query}'")
            
            elif any(month in search_lower for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                # User is searching for month names
                month_mapping = {
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                }
                for month_name, month_num in month_mapping.items():
                    if month_name in search_lower:
                        date_mask = filtered_df['date'].dt.month == month_num
                        search_mask = search_mask | date_mask
                        break
                st.info(f"💡 Searching for month and text containing '{search_query}'")
            
            # Apply search filter
            filtered_df = filtered_df[search_mask]
            
            if len(filtered_df) == 0:
                st.warning(f"No results found for '{search_query}'. Try different keywords or check your date filters.")
            else:
                st.success(f"Found {len(filtered_df)} transactions matching '{search_query}'")

        # Other filters
        st.subheader("🔍 Additional Filters")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            categories = ['All'] + list(df['category'].unique())
            selected_category = st.selectbox("Filter by Category", categories)
        with col2:
            transaction_types = ['All', 'credit', 'debit']
            selected_type = st.selectbox("Filter by Type", transaction_types)
        with col3:
            sources = ['All'] + list(df['source'].unique())
            selected_source = st.selectbox("Filter by Source", sources)
        with col4:
            sort_by = st.selectbox("Sort by", ['date', 'amount', 'category', 'description'])
        
        # Apply all filters
        filtered_df = df.copy()
        
        # Apply date filters
        if not df_with_dates.empty:
            if filter_type == "Specific Month" and 'selected_month' in locals() and selected_month:
                month_period = pd.Period(selected_month)
                filtered_df = filtered_df[filtered_df['date'].dt.to_period('M') == month_period]
            elif filter_type == "Date Range" and 'start_date' in locals() and 'end_date' in locals():
                filtered_df = filtered_df[
                    (filtered_df['date'].dt.date >= start_date) & 
                    (filtered_df['date'].dt.date <= end_date)
                ]
            elif filter_type == "Year" and 'selected_year' in locals() and selected_year:
                filtered_df = filtered_df[filtered_df['date'].dt.year == selected_year]
        
        # Apply other filters
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        if selected_type != 'All':
            filtered_df = filtered_df[filtered_df['transaction_type'] == selected_type]
        if selected_source != 'All':
            filtered_df = filtered_df[filtered_df['source'] == selected_source]
        
        # Sort data
        if sort_by == 'date':
            filtered_df = filtered_df.sort_values(by='date', ascending=False)
        else:
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)
        
        st.write(f"Showing {len(filtered_df)} of {len(df)} transactions")
        
        if view_type == "Monthly Summary":
            st.subheader("📊 Monthly Breakdown")
            
            if not df_with_dates.empty:
                # Create monthly summary
                monthly_data = df_with_dates.copy()
                monthly_data['month'] = monthly_data['date'].dt.to_period('M')
                monthly_summary = monthly_data.groupby('month').agg({
                    'amount': ['count', 'sum'],
                    'category': lambda x: list(x.unique())
                }).round(2)
                
                monthly_summary.columns = ['Transaction Count', 'Net Amount', 'Categories']
                monthly_summary['Income'] = monthly_data[monthly_data['amount'] > 0].groupby('month')['amount'].sum()
                monthly_summary['Expenses'] = monthly_data[monthly_data['amount'] < 0].groupby('month')['amount'].sum().abs()
                monthly_summary = monthly_summary.fillna(0)
                
                # Display monthly cards
                for month in monthly_summary.index:
                    with st.expander(f"📅 {month} - Net: ${monthly_summary.loc[month, 'Net Amount']:,.2f}"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Transactions", int(monthly_summary.loc[month, 'Transaction Count']))
                        with col2:
                            st.metric("Income", f"${monthly_summary.loc[month, 'Income']:,.2f}")
                        with col3:
                            st.metric("Expenses", f"${monthly_summary.loc[month, 'Expenses']:,.2f}")
                        with col4:
                            st.metric("Net", f"${monthly_summary.loc[month, 'Net Amount']:,.2f}")
                        
                        # Show transactions for this month
                        month_transactions = monthly_data[monthly_data['month'] == month]
                        st.write(f"**Categories used:** {', '.join(monthly_summary.loc[month, 'Categories'])}")
                        
                        if st.button(f"View {month} Details", key=f"view_{month}"):
                            st.dataframe(
                                month_transactions[['date', 'description', 'merchant', 'category', 'amount']].sort_values('date', ascending=False),
                                use_container_width=True
                            )
            else:
                st.info("No date information available for monthly breakdown")
        
        elif view_type == "Table View":
            # Table view with editable data
            st.subheader("📊 Data Table")
            
            # Format the dataframe for better display
            display_df = filtered_df.copy()
            if 'date' in display_df.columns:
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            
            # Reorder columns for better display
            column_order = ['date', 'description', 'merchant', 'category', 'amount', 'transaction_type', 'source']
            display_columns = [col for col in column_order if col in display_df.columns]
            display_df = display_df[display_columns]
            
            # Format amount column with color coding
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                    "amount": st.column_config.NumberColumn(
                        "Amount ($)",
                        format="$%.2f",
                        help="Positive = Income, Negative = Expense"
                    ),
                    "description": st.column_config.TextColumn("Description", width="large"),
                    "merchant": st.column_config.TextColumn("Merchant"),
                    "category": st.column_config.SelectboxColumn("Category"),
                    "transaction_type": st.column_config.SelectboxColumn("Type"),
                    "source": st.column_config.TextColumn("Source")
                }
            )
            
            # Add transaction management options
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Total Transactions:** {len(filtered_df)}")
            with col2:
                if st.button("🔄 Refresh Data"):
                    st.rerun()
            with col3:
                if st.button("📋 Copy Table to Clipboard"):
                    csv_data = filtered_df.to_csv(index=False)
                    st.write("Data ready to copy:")
                    st.code(csv_data[:500] + "..." if len(csv_data) > 500 else csv_data)
        
        else:  # Card View
            st.subheader("🃏 Card View")
            num_transactions = st.slider("Number to show", 5, min(50, len(filtered_df)), 10)
            recent_df = filtered_df.head(num_transactions)
            
            for _, row in recent_df.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{row['description']}**")
                        st.caption(f"🏪 {row['merchant']} | 📂 {row['category']}")
                        st.caption(f"📅 {row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/A'} | 📄 {row['source']}")
                    with col2:
                        if row['amount'] > 0:
                            st.success(f"+${row['amount']:.2f}")
                        else:
                            st.error(f"-${abs(row['amount']):.2f}")
                    with col3:
                        st.caption(f"Type: {row['transaction_type']}")
                        # Add delete button for individual transactions
                        if st.button("🗑️", key=f"delete_{row['id']}", help="Delete this transaction"):
                            delete_transaction(row['id'])
                            st.success("Transaction deleted!")
                            st.rerun()
                st.divider()
        
        # Summary stats for filtered data
        st.subheader("📈 Filtered Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        filtered_income = filtered_df[filtered_df['amount'] > 0]['amount'].sum()
        filtered_expenses = abs(filtered_df[filtered_df['amount'] < 0]['amount'].sum())
        filtered_net = filtered_income - filtered_expenses
        
        with col1:
            st.metric("Filtered Transactions", len(filtered_df))
        with col2:
            st.metric("Filtered Income", f"${filtered_income:,.2f}")
        with col3:
            st.metric("Filtered Expenses", f"${filtered_expenses:,.2f}")
        with col4:
            st.metric("Filtered Net", f"${filtered_net:,.2f}")
    
    else:
        st.info("No transactions yet. Upload a PDF, CSV, or add transactions manually to get started!")

    # Analytics Dashboard
    st.markdown("---")
    st.header("📊 Analytics Dashboard")

    if not st.session_state['transactions'].empty:
        df = st.session_state['transactions']

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_income = df[df['amount'] > 0]['amount'].sum()
        total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
        net_amount = total_income - total_expenses
        
        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            st.metric("Total Income", f"${total_income:,.2f}")
        with col3:
            st.metric("Total Expenses", f"${total_expenses:,.2f}")
        with col4:
            st.metric("Net Amount", f"${net_amount:,.2f}", delta=f"${net_amount:,.2f}")

        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'category' in df.columns:
                cat_summary = df.groupby('category')['amount'].sum().abs()
                fig = px.pie(
                    values=cat_summary.values, 
                    names=cat_summary.index, 
                    title="💸 Spending by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Income vs Expenses
            income_expense_data = pd.DataFrame({
                'Type': ['Income', 'Expenses'],
                'Amount': [total_income, total_expenses],
                'Color': ['green', 'red']
            })
            fig2 = px.bar(
                income_expense_data, 
                x='Type', 
                y='Amount', 
                color='Color',
                title="💰 Income vs Expenses",
                color_discrete_map={'green': '#00ff00', 'red': '#ff0000'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Transaction timeline
        if 'date' in df.columns and not df['date'].isna().all():
            df_with_dates = df.dropna(subset=['date'])
            if not df_with_dates.empty:
                # Monthly breakdown chart
                monthly_data = df_with_dates.copy()
                monthly_data['month'] = monthly_data['date'].dt.to_period('M').astype(str)
                monthly_summary = monthly_data.groupby('month').agg({
                    'amount': 'sum'
                }).reset_index()
                monthly_summary['income'] = monthly_data[monthly_data['amount'] > 0].groupby('month')['amount'].sum().values
                monthly_summary['expenses'] = monthly_data[monthly_data['amount'] < 0].groupby('month')['amount'].sum().abs().values
                monthly_summary = monthly_summary.fillna(0)
                
                fig3 = px.bar(
                    monthly_summary, 
                    x='month', 
                    y=['income', 'expenses'],
                    title="💰 Monthly Income vs Expenses",
                    labels={'value': 'Amount ($)', 'month': 'Month'},
                    barmode='group'
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                # Daily summary line chart
                daily_summary = df_with_dates.groupby(df_with_dates['date'].dt.date)['amount'].sum().reset_index()
                fig4 = px.line(
                    daily_summary, 
                    x='date', 
                    y='amount', 
                    title="📈 Daily Net Flow",
                    markers=True
                )
                st.plotly_chart(fig4, use_container_width=True)
        
        # Data export
        st.subheader("💾 Export Data")
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv,
                file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        with col2:
            if st.button("🗑️ Clear All Data"):
                clear_all_transactions()
                st.success("All data cleared!")
                st.rerun()

    else:
        st.info("📈 Upload some transactions to see analytics!")


if __name__ == "__main__":
    main()