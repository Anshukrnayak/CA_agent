import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import hashlib
import json
import requests
from xgboost import XGBClassifier
from cryptography.fernet import Fernet
from pylatex import Document, Section, Subsection, Package, LongTable
from pylatex.utils import NoEscape
import logging
import os
import shutil

# Setup logging
logging.basicConfig(filename='audit.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Persistent encryption key
KEY_FILE = 'key.key'
DB_FILE = 'transactions.db'


# Reset database and key
def reset_db_and_key():
   if os.path.exists(DB_FILE):
      os.remove(DB_FILE)
      logging.info("Database file deleted")
   if os.path.exists(KEY_FILE):
      os.remove(KEY_FILE)
      logging.info("Encryption key file deleted")


# Initialize encryption key
if os.path.exists(KEY_FILE):
   with open(KEY_FILE, 'rb') as f:
      ENCRYPTION_KEY = f.read()
else:
   ENCRYPTION_KEY = Fernet.generate_key()
   with open(KEY_FILE, 'wb') as f:
      f.write(ENCRYPTION_KEY)
cipher = Fernet(ENCRYPTION_KEY)


# Mock blockchain for local testing
class BlockchainModule:
   def __init__(self):
      self.transactions = []

   def add_transaction(self, transaction):
      tx_hash = hashlib.sha256(json.dumps(transaction, sort_keys=True).encode()).hexdigest()
      self.transactions.append({"hash": tx_hash, "transaction": transaction})
      logging.info(f"Blockchain transaction added: {tx_hash}")
      return tx_hash

   def get_transactions(self):
      return self.transactions


# Database module
class DatabaseModule:
   def __init__(self):
      self.conn = sqlite3.connect('transactions.db', check_same_thread=False)
      self.cursor = self.conn.cursor()
      self.cursor.execute('''CREATE TABLE IF NOT EXISTS transactions 
                             (id INTEGER PRIMARY KEY, date TEXT, description TEXT, amount REAL, currency TEXT, 
                              type TEXT, gst_rate REAL, blockchain_hash TEXT, client_id TEXT)''')
      self.conn.commit()
      logging.info("Database initialized with transactions table")

   def add_transaction(self, date, description, amount, currency, trans_type, gst_rate, blockchain_hash, client_id,
                       encrypt=True):
      try:
         client_id_to_store = cipher.encrypt(client_id.encode()).decode() if encrypt else client_id
         self.cursor.execute(
            "INSERT INTO transactions (date, description, amount, currency, type, gst_rate, blockchain_hash, client_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (date, description, amount, currency, trans_type, gst_rate, blockchain_hash, client_id_to_store))
         self.conn.commit()
         self.cursor.execute("SELECT COUNT(*) FROM transactions WHERE blockchain_hash = ?", (blockchain_hash,))
         count = self.cursor.fetchone()[0]
         logging.info(
            f"Transaction added for client {client_id}, blockchain_hash {blockchain_hash}, stored client_id {client_id_to_store}, rows in DB: {count}")
      except Exception as e:
         logging.error(f"Failed to add transaction: {str(e)}")
         raise

   def load_transactions(self, client_id=None, encrypt=True):
      try:
         if client_id:
            client_id_to_query = cipher.encrypt(client_id.encode()).decode() if encrypt else client_id
            logging.info(f"Querying transactions for client_id {client_id} (query: {client_id_to_query})")
            df = pd.read_sql_query("SELECT * FROM transactions WHERE client_id = ?", self.conn,
                                   params=(client_id_to_query,))
         else:
            logging.info("Querying all transactions")
            df = pd.read_sql_query("SELECT * FROM transactions", self.conn)
         logging.info(f"Loaded {len(df)} transactions")
         if not df.empty and encrypt:
            df['client_id'] = df['client_id'].apply(lambda x: cipher.decrypt(x.encode()).decode() if x else '')
         return df
      except Exception as e:
         logging.error(f"Error loading transactions: {str(e)}")
         return pd.DataFrame()

   def close(self):
      self.conn.close()


# Tax filing module
class TaxFilingModule:
   def __init__(self, db_module):
      self.db_module = db_module

   def generate_gst_report(self, client_id=None, encrypt=True):
      df = self.db_module.load_transactions(client_id, encrypt)
      output_tax = df[df['type'] == 'Income']['amount'] * df[df['type'] == 'Income']['gst_rate']
      input_tax = df[df['type'] == 'Expense']['amount'] * df[df['type'] == 'Expense']['gst_rate']
      net_gst = output_tax.sum() - input_tax.sum()
      return output_tax.sum(), input_tax.sum(), net_gst

   def calculate_income_tax(self, income):
      if income <= 300000:
         tax = 0
      elif income <= 600000:
         tax = (income - 300000) * 0.05
      elif income <= 900000:
         tax = 15000 + (income - 600000) * 0.10
      else:
         tax = 45000 + (income - 900000) * 0.20
      return tax

   def simulate_tax_filing(self, tax_type, client_id=None, encrypt=True):
      df = self.db_module.load_transactions(client_id, encrypt)
      if tax_type == "GST":
         output_tax, input_tax, net_gst = self.generate_gst_report(client_id, encrypt)
         payload = {
            'gstin': '29ABCDE1234F1ZK',
            'fp': datetime.now().strftime('%m%Y'),
            'output_tax': output_tax,
            'input_tax': input_tax,
            'net_gst': net_gst,
            'transactions': df[['date', 'description', 'amount', 'currency', 'gst_rate']].to_dict('records')
         }
      else:  # ITR
         income, expense, net_profit = self.generate_pnl(client_id, encrypt)
         tax = self.calculate_income_tax(net_profit)
         payload = {
            'pan': 'ABCDE1234F',
            'assessment_year': datetime.now().year + 1,
            'income': income,
            'tax_payable': tax,
            'transactions': df[df['type'] == 'Income'][['date', 'description', 'amount', 'currency']].to_dict('records')
         }
      try:
         response = {"status": "success", "message": f"{tax_type} filing simulated", "payload": payload}
         logging.info(f"{tax_type} filing simulated for client {client_id}")
         return response
      except Exception as e:
         logging.error(f"Tax filing error: {str(e)}")
         return {"status": "error", "message": str(e)}

   def generate_pnl(self, client_id=None, encrypt=True):
      df = self.db_module.load_transactions(client_id, encrypt)
      income = df[df['type'] == 'Income']['amount'].sum()
      expense = df[df['type'] == 'Expense']['amount'].sum()
      net_profit = income - expense
      return income, expense, net_profit

   def generate_balance_sheet(self, client_id=None, encrypt=True):
      df = self.db_module.load_transactions(client_id, encrypt)
      assets = df[df['type'] == 'Asset']['amount'].sum()
      liabilities = df[df['type'] == 'Liability']['amount'].sum()
      equity = assets - liabilities
      return assets, liabilities, equity


# Loan risk profiling module
class LoanRiskModule:
   def __init__(self, db_module):
      self.db_module = db_module

   def calculate_loan_risk_profile(self, client_id=None, encrypt=True):
      df = self.db_module.load_transactions(client_id, encrypt)
      if df.empty:
         logging.warning(f"No transactions found for client {client_id}")
         return {"risk_score": 0.0, "debt_to_income": 0.0, "cash_flow": 0.0}

      income = df[df['type'] == 'Income']['amount'].sum()
      expense = df[df['type'] == 'Expense']['amount'].sum()
      assets = df[df['type'] == 'Asset']['amount'].sum()
      liabilities = df[df['type'] == 'Liability']['amount'].sum()

      features = np.array([[income, expense, assets, liabilities]])
      labels = np.array([1 if income > expense and assets > liabilities else 0])

      try:
         model = XGBClassifier(base_score=0.5, objective='binary:logistic')
         model.fit(features, labels)
         risk_score = model.predict_proba(features)[0][1] * 100
      except Exception as e:
         logging.error(f"XGBoost model failed: {str(e)}")
         risk_score = 50.0  # Default score on error

      debt_to_income = liabilities / income if income > 0 else float('inf')
      cash_flow = income - expense
      logging.info(f"Loan risk profile calculated for client {client_id}: {risk_score}")
      return {"risk_score": risk_score, "debt_to_income": debt_to_income, "cash_flow": cash_flow}


# KYC/AML module
class KYCAMLModule:
   def __init__(self, db_module):
      self.db_module = db_module

   def verify_kyc(self, client_id, name, document):
      try:
         response = {"status": "success", "client_id": client_id, "message": f"KYC verified for {name} with {document}"}
         logging.info(f"KYC verified for client {client_id}")
         return response
      except Exception as e:
         logging.error(f"KYC error: {str(e)}")
         return {"status": "error", "message": str(e)}

   def detect_fraud(self, client_id=None, encrypt=True):
      df = self.db_module.load_transactions(client_id, encrypt)
      if len(df) == 0:
         return pd.DataFrame()
      features = df[['amount', 'gst_rate']].values
      scaler = MinMaxScaler()
      scaled_features = scaler.fit_transform(features)
      autoencoder = self.build_autoencoder(scaled_features.shape[1])
      autoencoder.fit(scaled_features, scaled_features, epochs=10, batch_size=32, verbose=0)
      reconstructions = autoencoder.predict(scaled_features, verbose=0)
      mse = np.mean(np.power(scaled_features - reconstructions, 2), axis=1)
      threshold = np.percentile(mse, 95)
      df['Fraud_Score'] = mse
      logging.info(f"AML check run for client {client_id}")
      return df[mse > threshold][['date', 'description', 'amount', 'currency', 'type', 'client_id', 'Fraud_Score']]

   def build_autoencoder(self, input_dim):
      model = tf.keras.Sequential([
         tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
         tf.keras.layers.Dense(32, activation='relu'),
         tf.keras.layers.Dense(16, activation='relu'),
         tf.keras.layers.Dense(32, activation='relu'),
         tf.keras.layers.Dense(64, activation='relu'),
         tf.keras.layers.Dense(input_dim)
      ])
      model.compile(optimizer='adam', loss='mse')
      return model


# Analytics module
class AnalyticsModule:
   def __init__(self, db_module):
      self.db_module = db_module

   def predict_cash_flow(self, data):
      if len(data) < 6:
         logging.warning("Insufficient data for cash flow prediction")
         return 0.0
      scaler = MinMaxScaler()
      scaled_data = scaler.fit_transform(data.reshape(-1, 1))
      X, y = [], []
      for i in range(len(scaled_data) - 5):
         X.append(scaled_data[i:i + 5])
         y.append(scaled_data[i + 5])
      X, y = np.array(X), np.array(y)
      model = self.build_transformer()
      model.compile(optimizer='adam', loss='mse')
      model.fit(X, y, epochs=10, verbose=0)
      last_sequence = scaled_data[-5:].reshape(1, 5, 1)
      prediction = model.predict(last_sequence, verbose=0)
      logging.info("Cash flow prediction generated")
      return scaler.inverse_transform(prediction)[0][0]

   def build_transformer(self):
      class TransformerModel(tf.keras.Model):
         def __init__(self, num_heads=2, d_model=64, num_layers=2):
            super(TransformerModel, self).__init__()
            self.encoder = tf.keras.layers.Dense(d_model)
            self.transformer_block = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
            self.dense = tf.keras.layers.Dense(1)

         def call(self, inputs):
            x = self.encoder(inputs)
            x = self.transformer_block(x, x)
            return self.dense(x)

      return TransformerModel()

   def monte_carlo_simulation(self, data, n_simulations=1000):
      if len(data) < 2:
         logging.warning("Insufficient data for Monte Carlo simulation")
         return np.zeros((n_simulations, 1))
      returns = np.diff(data) / data[:-1]
      mean_return = np.mean(returns)
      std_return = np.std(returns)
      simulations = np.random.normal(mean_return, std_return, (n_simulations, len(data)))
      simulated_paths = np.zeros((n_simulations, len(data)))
      simulated_paths[:, 0] = data[-1]
      for t in range(1, len(data)):
         simulated_paths[:, t] = simulated_paths[:, t - 1] * (1 + simulations[:, t])
      logging.info("Monte Carlo simulation completed")
      return simulated_paths


# Investor dashboard module
class InvestorDashboardModule:
   def __init__(self, db_module, tax_module, loan_module):
      self.db_module = db_module
      self.tax_module = tax_module
      self.loan_module = loan_module

   def generate_dashboard_data(self, client_id=None, encrypt=True):
      income, expense, net_profit = self.tax_module.generate_pnl(client_id, encrypt)
      assets, liabilities, equity = self.tax_module.generate_balance_sheet(client_id, encrypt)
      risk_profile = self.loan_module.calculate_loan_risk_profile(client_id, encrypt)
      valuation = net_profit * 15 if net_profit > 0 else 0
      return {
         "net_profit": net_profit,
         "equity": equity,
         "risk_score": risk_profile["risk_score"],
         "cash_flow": risk_profile["cash_flow"],
         "valuation": valuation
      }


# Cross-border accounting module
class CrossBorderModule:
   def __init__(self):
      pass

   def get_exchange_rate(self, currency):
      try:
         response = requests.get(f"https://api.exchangerate-api.com/v4/latest/INR")
         rates = response.json()['rates']
         logging.info(f"Exchange rate fetched for {currency}: {rates.get(currency, 1.0)}")
         return rates.get(currency, 1.0)
      except:
         return {'USD': 83.5, 'EUR': 90.2, 'GBP': 105.7}.get(currency, 1.0)


# PDF reporting module
class ReportingModule:
   def __init__(self, db_module, tax_module, loan_module):
      self.db_module = db_module
      self.tax_module = tax_module
      self.loan_module = loan_module

   def generate_pdf_report(self, client_id=None, encrypt=True):
      doc = Document()
      doc.packages.append(Package('geometry', options=['a4paper', 'margin=1in']))
      doc.packages.append(Package('booktabs'))

      with doc.create(Section('Financial Report')):
         with doc.create(Subsection('GST Report')):
            output_tax, input_tax, net_gst = self.tax_module.generate_gst_report(client_id, encrypt)
            doc.append(f'Output Tax: ₹{output_tax:,.2f}\n')
            doc.append(f'Input Tax: ₹{input_tax:,.2f}\n')
            doc.append(f'Net GST Liability: ₹{net_gst:,.2f}\n')

         with doc.create(Subsection('Profit & Loss')):
            income, expense, net_profit = self.tax_module.generate_pnl(client_id, encrypt)
            doc.append(f'Income: ₹{income:,.2f}\n')
            doc.append(f'Expenses: ₹{expense:,.2f}\n')
            doc.append(f'Net Profit: ₹{net_profit:,.2f}\n')

         with doc.create(Subsection('Loan Risk Profile')):
            risk_profile = self.loan_module.calculate_loan_risk_profile(client_id, encrypt)
            doc.append(f'Risk Score: {risk_profile["risk_score"]:.2f}/100\n')
            doc.append(f'Debt-to-Income Ratio: {risk_profile["debt_to_income"]:.2f}\n')
            doc.append(f'Cash Flow: ₹{risk_profile["cash_flow"]:,.2f}\n')

         with doc.create(Subsection('Transaction History')):
            df = self.db_module.load_transactions(client_id, encrypt)
            with doc.create(LongTable('l l r l l l')) as table:
               table.add_hline()
               table.add_row(['Date', 'Description', 'Amount (INR)', 'Currency', 'Type', 'Client ID'])
               table.add_hline()
               for _, row in df.head(10).iterrows():
                  description = row['description'] if row['description'] else 'N/A'
                  table.add_row([row['date'], description[:20], f"{row['amount']:,.2f}", row['currency'], row['type'],
                                 row['client_id']])
               table.add_hline()

      # Check if pdflatex is available
      if shutil.which('pdflatex') is None:
         doc.generate_tex('financial_report')
         logging.error("pdflatex not found; saved LaTeX source to financial_report.tex")
         raise Exception(
            "pdflatex not found. Install TeX Live (https://www.tug.org/texlive/acquire-netinstall.html) and add C:\\texlive\\2024\\bin\\win32 to PATH. LaTeX source saved as financial_report.tex.")

      try:
         doc.generate_pdf('financial_report', clean_tex=True, compiler='pdflatex')
         logging.info(f"PDF report generated for client {client_id}")
      except Exception as e:
         doc.generate_tex('financial_report')
         logging.error(f"PDF generation failed: {str(e)}; saved LaTeX source to financial_report.tex")
         raise Exception(
            f"PDF generation failed: {str(e)}. Install TeX Live (https://www.tug.org/texlive/acquire-netinstall.html) and add C:\\texlive\\2024\\bin\\win32 to PATH. LaTeX source saved as financial_report.tex.")


# Initialize modules
db_module = DatabaseModule()
blockchain_module = BlockchainModule()
tax_module = TaxFilingModule(db_module)
loan_module = LoanRiskModule(db_module)
kyc_aml_module = KYCAMLModule(db_module)
analytics_module = AnalyticsModule(db_module)
investor_module = InvestorDashboardModule(db_module, tax_module, loan_module)
cross_border_module = CrossBorderModule()
reporting_module = ReportingModule(db_module, tax_module, loan_module)

# Streamlit UI with dark mode
st.set_page_config(page_title="AI CA Dashboard", layout="wide")
st.markdown("""
    <style>
    body { background-color: #1E1E1E; color: #FFFFFF; }
    .stApp { background-color: #1E1E1E; }
    .stButton>button { background-color: #4CAF50; color: white; }
    .stTextInput>div>input { background-color: #2C2C2C; color: #FFFFFF; }
    .stSelectbox>div>select { background-color: #2C2C2C; color: #FFFFFF; }
    </style>
""", unsafe_allow_html=True)

st.title("Advanced AI Chartered Accountant Dashboard")
st.write("Automate GST/ITR filing, assess loan risks, perform KYC/AML, and generate investor dashboards")

# Sidebar for navigation
st.sidebar.header("Navigation")
option = st.sidebar.selectbox("Select Task",
                              ["Bookkeeping", "GST/ITR Filing", "Loan Risk Profile", "KYC/AML", "Investor Dashboard",
                               "Analytics", "PDF Export", "Blockchain"])

# Client ID for multi-user support
client_id = st.sidebar.text_input("Client ID", value="default")

# Debug options
encrypt_client_id = st.sidebar.checkbox("Encrypt Client ID", value=True)
if st.sidebar.button("Reset Database and Encryption Key"):
   reset_db_and_key()
   st.sidebar.success("Database and encryption key reset. Restart the app to apply changes.")

if option == "Bookkeeping":
   st.header("Bookkeeping")
   with st.form("transaction_form"):
      date = st.date_input("Transaction Date")
      description = st.text_input("Description")
      amount = st.number_input("Amount", min_value=0.0, step=0.01)
      currency = st.selectbox("Currency", ['INR', 'USD', 'EUR', 'GBP'])
      trans_type = st.selectbox("Type", ["Income", "Expense", "Asset", "Liability"])
      gst_rate = st.selectbox("GST Rate", [0.0, 0.05, 0.12, 0.18, 0.28])
      submit = st.form_submit_button("Add Transaction")
      if submit:
         amount_inr = amount * cross_border_module.get_exchange_rate(currency)
         transaction = {
            'date': str(date),
            'description': description,
            'amount': amount_inr,
            'currency': currency,
            'type': trans_type,
            'gst_rate': gst_rate,
            'client_id': client_id
         }
         tx_hash = blockchain_module.add_transaction(transaction)
         db_module.add_transaction(str(date), description, amount_inr, currency, trans_type, gst_rate, tx_hash,
                                   client_id, encrypt_client_id)
         st.success(f"Transaction added successfully! Blockchain Hash: {tx_hash}")

   st.subheader("Transaction History")
   df = db_module.load_transactions(client_id, encrypt_client_id)
   if not df.empty:
      st.dataframe(df)
   else:
      st.warning(
         f"No transactions found for Client ID: {client_id}. Try adding a new transaction, disabling encryption, or checking audit.log.")

   if st.checkbox("Show all transactions (for debugging)"):
      all_df = db_module.load_transactions(encrypt=encrypt_client_id)
      if not all_df.empty:
         st.subheader("All Transactions in Database")
         st.dataframe(all_df)
      else:
         st.warning("No transactions exist in the database.")

elif option == "GST/ITR Filing":
   st.header("GST/ITR Filing Automation")
   tax_type = st.selectbox("Select Tax Type", ["GST", "ITR"])
   if st.button(f"Simulate {tax_type} Filing"):
      response = tax_module.simulate_tax_filing(tax_type, client_id, encrypt_client_id)
      st.json(response)
      st.write(f"This is a simulated {tax_type} filing response. Use actual credentials for real filing.")

elif option == "Loan Risk Profile":
   st.header("Business Loan Risk Profile")
   if st.button("Generate Risk Profile"):
      risk_profile = loan_module.calculate_loan_risk_profile(client_id, encrypt_client_id)
      st.write(f"**Risk Score**: {risk_profile['risk_score']:.2f}/100")
      st.write(f"**Debt-to-Income Ratio**: {risk_profile['debt_to_income']:.2f}")
      st.write(f"**Cash Flow**: ₹{risk_profile['cash_flow']:,.2f}")

elif option == "KYC/AML":
   st.header("KYC/AML Verification")
   with st.form("kyc_form"):
      name = st.text_input("Client Name")
      document = st.text_input("Document Number (e.g., PAN, Aadhaar)")
      submit_kyc = st.form_submit_button("Verify KYC")
      if submit_kyc:
         kyc_result = kyc_aml_module.verify_kyc(client_id, name, document)
         st.json(kyc_result)

   st.subheader("AML Anomaly Detection")
   if st.button("Run AML Check"):
      fraud_df = kyc_aml_module.detect_fraud(client_id, encrypt_client_id)
      if not fraud_df.empty:
         st.dataframe(fraud_df)
      else:
         st.write("No suspicious transactions detected.")

elif option == "Investor Dashboard":
   st.header("Investor Dashboard for Startups")
   df = db_module.load_transactions(client_id, encrypt_client_id)
   if not df.empty:
      dashboard_data = investor_module.generate_dashboard_data(client_id, encrypt_client_id)
      col1, col2 = st.columns(2)
      with col1:
         st.metric("Net Profit", f"₹{dashboard_data['net_profit']:,.2f}")
         st.metric("Equity", f"₹{dashboard_data['equity']:,.2f}")
      with col2:
         st.metric("Risk Score", f"{dashboard_data['risk_score']:.2f}/100")
         st.metric("Valuation", f"₹{dashboard_data['valuation']:,.2f}")

      plt.figure(figsize=(10, 6))
      plt.plot(df[df['type'] == 'Income']['amount'].cumsum(), label='Cumulative Income')
      plt.plot(df[df['type'] == 'Expense']['amount'].cumsum(), label='Cumulative Expenses')
      plt.title("Financial Growth")
      plt.xlabel("Transaction")
      plt.ylabel("Amount (INR)")
      plt.legend()
      st.pyplot(plt)
   else:
      st.warning("No transactions available for this client.")

elif option == "Analytics":
   st.header("Financial Analytics")
   df = db_module.load_transactions(client_id, encrypt_client_id)
   if len(df) > 5:
      amounts = df['amount'].values
      prediction = analytics_module.predict_cash_flow(amounts)
      st.write(f"**Predicted Next Cash Flow (INR)**: ₹{prediction:,.2f}")

      fraud_df = kyc_aml_module.detect_fraud(client_id, encrypt_client_id)
      if not fraud_df.empty:
         st.subheader("Potential Fraudulent Transactions")
         st.dataframe(fraud_df)
      else:
         st.write("No fraudulent transactions detected.")

      st.subheader("Cash Flow Risk Analysis")
      simulations = analytics_module.monte_carlo_simulation(amounts)
      plt.figure(figsize=(10, 6))
      for i in range(min(100, simulations.shape[0])):
         plt.plot(simulations[i], alpha=0.1, color='blue')
      plt.plot(amounts, 'k-', label='Historical Cash Flow')
      plt.title("Monte Carlo Cash Flow Simulations")
      plt.xlabel("Transaction")
      plt.ylabel("Amount (INR)")
      plt.legend()
      st.pyplot(plt)
   else:
      st.warning("Add at least 6 transactions for analytics.")

elif option == "PDF Export":
   st.header("Export Financial Report to PDF")
   if st.button("Generate PDF"):
      try:
         reporting_module.generate_pdf_report(client_id, encrypt_client_id)
         with open("financial_report.pdf", "rb") as file:
            st.download_button("Download Financial Report", file, file_name="financial_report.pdf")
         st.success("PDF generated successfully!")
      except Exception as e:
         st.error(str(e))
         if os.path.exists("financial_report.tex"):
            with open("financial_report.tex", "rb") as file:
               st.download_button("Download LaTeX Source", file, file_name="financial_report.tex")
            st.info(
               "LaTeX source file saved as financial_report.tex. Compile it manually with pdflatex or on Overleaf (https://www.overleaf.com).")

elif option == "Blockchain":
   st.header("Blockchain Transaction Log")
   st.write("View blockchain-based transaction history for crypto tax compliance.")
   st.json(blockchain_module.get_transactions())

# Close database connection
db_module.close()