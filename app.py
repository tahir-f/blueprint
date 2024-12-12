import streamlit as st
import time
import lightgbm as lgb
import xgboost as xgb

# Load Models
xgb_model = xgb.Booster(model_file="xgboost_model.json")
lgb_model = lgb.Booster(model_file="lightgbm_model.txt")

# Streamed response generator
def response_generator(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.inputs = {
        "SquareFeet": None,
        "Bedrooms": None,
        "Bathrooms": None,
        "Wage": None,
        "MaterialGrade": None,
        "Floors": None,
    }
    st.session_state.current_step = 0
    st.session_state.last_question = None

# Reset app state
def reset_app():
    st.session_state.messages = []
    st.session_state.inputs = {
        "SquareFeet": None,
        "Bedrooms": None,
        "Bathrooms": None,
        "Wage": None,
        "MaterialGrade": None,
        "Floors": None,
    }
    st.session_state.current_step = 0
    st.session_state.last_question = None

# Add custom CSS for styling
st.markdown(
    """
    <style>
        /* General Body Styling */
        body {
            font-family: 'Roboto', sans-serif;
            color: #002082;
            background-color: ghostwhite;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #002082;
            color: white;
        }

        [data-testid="stSidebar"] button {
            background-color: transparent !important;
            color: white !important;
            border: 2px solid white !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            padding: 6px 12px !important;
        }

        [data-testid="stSidebar"] button:hover {
            background-color: white !important;
            color: #002082 !important;
        }

        /* Sidebar Logo Styling */
        #sidebar-logo {
            margin-bottom: 20px;
            text-align: center;
        }

        #sidebar-logo img {
            max-width: 100%;
        }

        /* Chat Message Styling */
        .stChatMessageUser {
            background-color: #CED8F7;
            border-radius: 10px;
            padding: 10px;
            color: #002082;
        }

        .stChatMessageAssistant {
            background-color: #002082;
            color: white;
            border-radius: 10px;
            padding: 10px;
        }

        .stChatMessageUser .stMessageIcon svg {
            fill: #002082;
            background-color: white;
            border-radius: 50%;
        }

        .stChatMessageAssistant .stMessageIcon {
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #002082;
            color: white;
            font-weight: bold;
            font-size: 20px;
            height: 40px;
            width: 40px;
            border-radius: 50%;
        }

        .stChatMessageAssistant .stMessageIcon::after {
            content: "B";
        }

        /* Streamlit Default Accent Color Overwrite */
        :root {
            --streamlit-accent-color: #002082;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add logo to the sidebar
with st.sidebar:
    st.markdown(
        """
        <div id="sidebar-logo">
            <img src="logo.png" alt="Blueprint.AI Logo">
        </div>
        """,
        unsafe_allow_html=True,
    )

# Sidebar options
st.sidebar.header("Options")
if st.sidebar.button("Reset Chat"):
    reset_app()

# Sidebar editable inputs
st.sidebar.header("Edit Inputs")
for key, value in st.session_state.inputs.items():
    if key in ["SquareFeet", "Bedrooms", "Bathrooms", "Floors"]:
        st.session_state.inputs[key] = st.sidebar.number_input(key, value=value or 0, step=1)
    elif key == "Wage":
        st.session_state.inputs[key] = st.sidebar.number_input(key, value=value or 0.0, step=0.1)
    elif key == "MaterialGrade":
        st.session_state.inputs[key] = st.sidebar.number_input(key, value=value or 1, min_value=1, max_value=3, step=1)

if st.sidebar.button("Update & Recalculate"):
    pass

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Questions for interactive flow
questions = [
    "How many square feet is your house?",
    "How many bedrooms does your house have?",
    "How many bathrooms?",
    "What is the labor cost per square foot (Wage)?",
    "What is the material grade? (1 - Low, 3 - High)",
    "How many floors will your house have?",
]

# Main logic for asking questions
if st.session_state.current_step < len(questions):
    current_question = questions[st.session_state.current_step]

    if st.session_state.last_question != current_question:
        with st.chat_message("assistant"):
            st.markdown("".join(response_generator(current_question)))
        st.session_state.messages.append({"role": "assistant", "content": current_question})
        st.session_state.last_question = current_question

    # Wait for user input
    user_response = st.chat_input("Your answer...")
    if user_response:
        st.session_state.messages.append({"role": "user", "content": user_response})

        # Map user input to the appropriate field
        key = list(st.session_state.inputs.keys())[st.session_state.current_step]
        if key in ["SquareFeet", "Bedrooms", "Bathrooms", "Floors"]:
            st.session_state.inputs[key] = int(user_response)
        elif key == "Wage":
            st.session_state.inputs[key] = float(user_response)
        elif key == "MaterialGrade":
            st.session_state.inputs[key] = int(user_response)

        # Advance to the next question
        st.session_state.current_step += 1
        st.rerun()

else:
    # All inputs are provided; generate prediction
    input_features = [
        st.session_state.inputs["SquareFeet"],
        st.session_state.inputs["Bedrooms"],
        st.session_state.inputs["Bathrooms"],
        st.session_state.inputs["Wage"],
        st.session_state.inputs["MaterialGrade"],
        st.session_state.inputs["Floors"],
    ]

    selected_model = st.sidebar.radio("Select Prediction Model", ["LightGBM", "XGBoost"], index=0)

    if selected_model == "XGBoost":
        dmatrix = xgb.DMatrix([input_features], feature_names=["SquareFeet", "Bedrooms", "Bathrooms", "Wage", "MaterialGrade", "Floors"])
        prediction = xgb_model.predict(dmatrix)[0]
    else:
        prediction = lgb_model.predict([input_features])[0]

    st.session_state.messages.append({"role": "assistant", "content": f"🏠 The estimated total cost is: ${round(prediction, 2)}"})
    with st.chat_message("assistant"):
        st.markdown(f"### 🏠 The estimated total cost is: **${round(prediction, 2)}**")

    breakdown = {
        "Base Cost": round(prediction * 0.5, 2),
        "Bedrooms": round(prediction * 0.1 * st.session_state.inputs["Bedrooms"], 2),
        "Bathrooms": round(prediction * 0.05 * st.session_state.inputs["Bathrooms"], 2),
        "Materials": round(prediction * 0.2 * st.session_state.inputs["MaterialGrade"], 2),
        "Floors": round(prediction * 0.15 * st.session_state.inputs["Floors"], 2),
    }

    with st.chat_message("assistant"):
        st.markdown("### Cost Breakdown")
        st.table(breakdown)
