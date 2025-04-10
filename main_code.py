import streamlit as st
import re

# Custom CSS with dark mode-inspired color scheme
st.markdown(
    """
    <style>
    :root {
        --primary: #4CAF50;  /* Vibrant green */
        --secondary: #2196F3;  /* Bright blue */
        --accent: #FFC107;  /* Amber accent */
        --dark-bg: #121212;  /* Very dark background */
        --card-bg: #1E1E1E;  /* Slightly lighter cards */
        --text-light: #FFFFFF;  /* White text */
        --text-muted: #B0B0B0;  /* Light gray for secondary text */
        --warning: #FF9800;  /* Orange warning */
        --danger: #F44336;  /* Red for errors */
    }

    body {
        background-color: var(--dark-bg);
        color: var(--text-light);
        font-family: 'Arial', sans-serif;
        line-height: 1.6;
    }
    .stApp {
        background-color: var(--dark-bg);
    }
    .sidebar .sidebar-content {
        background-color: #1A1A1A;
        border-right: 1px solid #333333;
    }
    .stTextArea, .stTextInput {
        background-color: var(--card-bg);
        color: var(--text-light);
        border-radius: 8px;
        border: 1px solid #333333;
        padding: 12px;
    }
    .stButton>button {
        background-color: var(--primary);
        color: var(--text-light);
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #43A047;
    }
    .stMarkdown {
        font-size: 16px;
        color: var(--text-light);
    }
    .stExpander {
        background-color: var(--card-bg);
        border-radius: 8px;
        border: 1px solid #333333;
        margin-bottom: 16px;
    }

    /* High contrast alert boxes */
    .warning-box {
        background-color: #332900;
        border-left: 5px solid var(--warning);
        padding: 16px;
        border-radius: 4px;
        margin-bottom: 16px;
        color: var(--text-light);
    }
    .success-box {
        background-color: #1E3A1E;
        border-left: 5px solid var(--primary);
        padding: 16px;
        border-radius: 4px;
        margin-bottom: 16px;
        color: var(--text-light);
    }
    .example-box {
        background-color: #252525;
        border-left: 5px solid var(--secondary);
        padding: 14px;
        border-radius: 4px;
        margin: 14px 0;
        color: var(--text-light);
    }
    .suggestion-box {
        background-color: #1A2A3A;
        border-left: 5px solid var(--secondary);
        padding: 16px;
        border-radius: 4px;
        margin: 16px 0;
        color: var(--text-light);
    }
    .term-highlight {
        background-color: var(--warning);
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
        color: #000000;
    }
    .sidebar-info {
        background-color: #252525;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333333;
        margin-top: 20px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary);
    }
    a {
        color: var(--secondary);
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        padding: 10px;
        color: var(--text-muted);
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Country-specific green marketing guidelines
compliance_data = {
    "UK": {
        "guideline": "Follow the Advertising Standards Authority (ASA) guidelines on environmental claims. Ensure all claims are clear, accurate, and substantiated.",
        "link": "https://www.asa.org.uk/type/non_broadcast/code_section/11.html",
        "icon": "🇬🇧"
    },
    "Netherlands": {
        "guideline": "Comply with the Netherlands Authority for Consumers and Markets (ACM) sustainability claims guidelines. Avoid vague terms and provide evidence for all environmental benefits.",
        "link": "https://www.acm.nl/en/publications/guidelines-sustainability-claims",
        "icon": "🇳🇱"
    },
    "USA": {
        "guideline": "Adhere to the Federal Trade Commission (FTC) Green Guides. Qualify general environmental benefit claims with specific environmental benefits.",
        "link": "https://www.ftc.gov/news-events/media-resources/truth-advertising/green-guides",
        "icon": "🇺🇸"
    },
    "EU": {
        "guideline": "Follow the EU Green Claims Directive. Ensure all environmental claims are truthful, accurate, and based on scientific evidence.",
        "link": "https://ec.europa.eu/commission/presscorner/detail/en/ip_23_1661",
        "icon": "🇪🇺"
    }
}

# Greenwashing terms with examples and alternatives
greenwashing_terms = {
    "eco-friendly": {
        "examples": [
            "H&M's 'Conscious Collection' was criticized for using this term while still producing fast fashion at scale.",
            "BP's 'Keep Advancing' campaign used this term while continuing fossil fuel extraction."
        ],
        "alternatives": ["Specify exact environmental benefits (e.g., 'made with 50% recycled materials')"]
    },
    "carbon-neutral": {
        "examples": [
            "Shell's 'Drive Carbon Neutral' program was criticized for relying on offsets rather than reducing emissions.",
            "Airlines using this term while expanding flight routes have faced backlash."
        ],
        "alternatives": ["Provide specific emission reduction figures and methods", "Clarify if this includes offsets"]
    },
    "sustainable": {
        "examples": [
            "Nestlé's 'Sustainable Packaging Initiative' faced criticism as most packaging remained non-recyclable.",
            "Fast fashion brands using this term while increasing production volumes."
        ],
        "alternatives": ["Define what aspect is sustainable (materials, production, etc.)",
                         "Provide measurable targets"]
    }
}


def check_greenwashing(text):
    """Check for potentially misleading environmental claims."""
    flagged_terms = []
    suggestions = []
    examples = []

    for term, data in greenwashing_terms.items():
        if re.search(fr'\b{term}\b', text, re.IGNORECASE):
            flagged_terms.append(term)
            suggestions.extend(data["alternatives"])
            examples.extend(data["examples"])

    if flagged_terms:
        warning_message = f"""
        <div class='warning-box'>
            <h3 style='color: var(--warning);'>⚠️ Potential Greenwashing Detected</h3>
            <p>The following terms may be misleading: <strong>{', '.join(flagged_terms)}</strong></p>
        </div>
        """

        examples_message = ""
        if examples:
            examples_message = "<h4 style='color: var(--secondary);'>Recent Controversial Examples:</h4>"
            for example in set(examples[:3]):
                examples_message += f"<div class='example-box'><p>{example}</p></div>"

        suggestions_message = ""
        if suggestions:
            unique_suggestions = list(set(suggestions))
            suggestions_message = f"""
            <div class='suggestion-box'>
                <h4 style='color: var(--secondary);'>Suggested Improvements:</h4>
                <ul>
                    {''.join([f'<li>{suggestion}</li>' for suggestion in unique_suggestions])}
                </ul>
            </div>
            """

        return warning_message + examples_message + suggestions_message

    return """
    <div class='success-box'>
        <h3 style='color: var(--primary);'>✅ Claim Appears Substantiated</h3>
        <p>Your marketing claim doesn't contain obvious greenwashing terms. For best practices:</p>
        <ul>
            <li>Ensure you have evidence to back all environmental claims</li>
            <li>Be specific about which aspects are sustainable</li>
            <li>Consider third-party certifications for added credibility</li>
        </ul>
    </div>
    """


def highlight_terms(text):
    """Highlight potential greenwashing terms in the text."""
    for term in greenwashing_terms.keys():
        text = re.sub(
            fr'\b({term})\b',
            r'<span class="term-highlight">\1</span>',
            text,
            flags=re.IGNORECASE
        )
    return text


# Main Dashboard Layout
st.title("🌿 Sustainable Marketing Compliance Dashboard")

# Sidebar for country selection with flag icon
st.sidebar.header("🌍 Select Your Market")
selected_country = st.sidebar.selectbox("Country/Region", list(compliance_data.keys()), format_func=lambda x: f"{compliance_data[x]['icon']} {x}")
st.sidebar.markdown(f"""
<div class='sidebar-info'>
    <h4 style='color: var(--primary);'>Compliance Guidelines for {selected_country}:</h4>
    <p>{compliance_data[selected_country]['guideline']}</p>
    <a href="{compliance_data[selected_country]['link']}" target="_blank">Read official guidelines →</a>
</div>
""", unsafe_allow_html=True)

# Main content tabs
tab1, tab2 = st.tabs(["📝 Campaign Checker", "📚 Case Studies"])

with tab1:
    st.header("Greenwashing Checker")
    user_text = st.text_area("Enter your marketing claim or slogan:", height=150,
                             placeholder="Example: Our product is 100% eco-friendly and sustainable...")

    if st.button("Analyze for Greenwashing", type="primary"):
        if not user_text or not user_text.strip():
            st.warning("Please enter some text to analyze")
        else:
            st.markdown("### Analysis Results", unsafe_allow_html=True)
            st.markdown(check_greenwashing(user_text), unsafe_allow_html=True)

            st.markdown("### Text Preview with Highlighted Terms", unsafe_allow_html=True)
            highlighted_text = highlight_terms(user_text) if user_text else "No text to analyze"
            st.markdown(
                f"""
                <div style='
                    border: 1px solid #333333;
                    padding: 20px;
                    border-radius: 8px;
                    background-color: var(--card-bg);
                    margin-top: 10px;
                '>
                    {highlighted_text}
                </div>
                """,
                unsafe_allow_html=True
            )

with tab2:
    st.header("Green Marketing Case Studies")

    with st.expander("✅ Positive Examples", expanded=True):
        st.markdown("""
        <div class='success-box'>
            <h3>Patagonia's Environmental Initiatives</h3>
            <p><strong>Why it works:</strong> Specific, measurable claims with full transparency</p>
            <ul>
                <li>"100% of our cotton is organic" - verifiable claim</li>
                <li>"1% of sales donated to environmental causes" - concrete commitment</li>
                <li>Worn Wear program for used gear - circular economy in action</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='success-box'>
            <h3>Seventh Generation Cleaning Products</h3>
            <p><strong>Why it works:</strong> Clear labeling with scientific backing</p>
            <ul>
                <li>"Plant-based ingredients (list provided)" - no vague "natural" claims</li>
                <li>"Biodegradable within 28 days in compliance with OECD 301B" - testable standard</li>
                <li>B Corp certification - third-party validation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("⚠️ Controversial Examples", expanded=True):
        st.markdown("""
        <div class='warning-box'>
            <h3>Volkswagen's 'Clean Diesel' Campaign</h3>
            <p><strong>Why it failed:</strong> Deliberate deception with technical claims</p>
            <ul>
                <li>Claimed low emissions while using defeat devices</li>
                <li>Fined $30 billion in the emissions scandal</li>
                <li>Resulted in complete campaign withdrawal</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='warning-box'>
            <h3>Fiji Water's 'Carbon Negative' Claim</h3>
            <p><strong>Issues:</strong> Questionable carbon accounting</p>
            <ul>
                <li>Claim didn't account for shipping emissions</li>
                <li>Offset program lacked transparency</li>
                <li>Class action lawsuit filed in 2023</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Made for the Future17 Program 2025
</div>
""", unsafe_allow_html=True)
