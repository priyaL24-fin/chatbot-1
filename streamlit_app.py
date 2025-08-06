import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("What is up?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
def calculate_risk_score(stock_data):
    volatility = stock_data['Close'].pct_change().std()
    beta = 1.2  # Placeholder, you can fetch from APIs like Alpha Vantage

    # Risk scoring logic (0-100)
    if volatility > 0.03:
        risk_level = "High"
        score = 80
    elif volatility > 0.02:
        risk_level = "Moderate"
        score = 50
    else:
        risk_level = "Low"
        score = 20

    return score, risk_level, volatility
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_chatgpt_about_risk(ticker, risk_score, risk_level, volatility):
    prompt = f"""
    Analyze the risk of investing in {ticker} stock. It has a historical volatility of {volatility:.2f} and a risk score of {risk_score}/100, which is considered {risk_level}.
    Explain what makes it risky or safe for investors, and suggest investor profiles suitable for it.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a financial risk analysis expert."},
            {"role": "user", "content": prompt}
        ]
    )

    return response['choices'][0]['message']['content']
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from risk_model import calculate_risk_score, ask_chatgpt_about_risk

st.set_page_config(page_title="RiskRadar AI", layout="wide")

st.title("📉 RiskRadar AI - Stock Risk Analysis Chatbot")

ticker = st.text_input("Enter a US Stock Ticker (e.g., AAPL, TSLA):")

if st.button("Analyze Risk") and ticker:
    with st.spinner("Fetching data and analyzing..."):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")

            if hist.empty:
                st.error("No data found for this ticker.")
            else:
                st.subheader("📈 Stock Price Chart")
                st.line_chart(hist['Close'])

                score, level, volatility = calculate_risk_score(hist)

                st.metric("📊 Risk Score", score)
                st.metric("⚠️ Risk Level", level)
                st.metric("📉 Volatility (6M)", f"{volatility:.2%}")

                explanation = ask_chatgpt_about_risk(ticker, score, level, volatility)
                st.subheader("🤖 GPT-4 Risk Explanation")
                st.markdown(explanation)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
class StockRiskAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.risk_factors = {}
        self.weights = {
            "volatility": 0.20,
            "financial": 0.25,
            "valuation": 0.15,
            "sentiment": 0.15,
            "sector": 0.10,
            "esg": 0.15
        }

    def input_risk_scores(self, volatility, financial, valuation, sentiment, sector, esg):
        self.risk_factors = {
            "volatility": volatility,
            "financial": financial,
            "valuation": valuation,
            "sentiment": sentiment,
            "sector": sector,
            "esg": esg
        }

    def calculate_risk_score(self):
        total_score = 0
        for factor, score in self.risk_factors.items():
            weight = self.weights.get(factor, 0)
            total_score += score * weight
        return round(total_score, 2)

    def get_risk_level(self):
        score = self.calculate_risk_score()
        if score <= 30:
            return "Low Risk"
        elif 30 < score <= 60:
            return "Moderate Risk"
        else:
            return "High Risk"

    def print_summary(self):
        print(f"📈 Risk Summary for {self.symbol}")
        for k, v in self.risk_factors.items():
            print(f"- {k.capitalize()} Risk: {v}/100")
        final_score = self.calculate_risk_score()
        print(f"\n🔢 Final Risk Score: {final_score}/100")
        print(f"🛑 Risk Level: {self.get_risk_level()}")



