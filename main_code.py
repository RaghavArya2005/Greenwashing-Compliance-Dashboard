import streamlit as st
from streamlit_option_menu import option_menu
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
from datetime import datetime

# Set page config must be first
st.set_page_config(layout="wide")

# Then load other components
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Levenshtein distance function for spelling variations
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


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

# Expanded greenwashing terms with embeddings and categories
greenwashing_terms = {
    # Vague sustainability terms
    "eco-friendly": {
        "examples": ["H&M's 'Conscious Collection'", "BP's 'Keep Advancing' campaign"],
        "alternatives": ["Specify exact environmental benefits (e.g., 'made with 50% recycled materials')"],
        "similar_phrases": ["environmentally friendly", "planet-friendly", "earth-conscious", "green", "eco-conscious"],
        "category": "Vague Claims"
    },
    "sustainable": {
        "examples": ["Nestlé's 'Sustainable Packaging'", "Fast fashion sustainability claims"],
        "alternatives": ["Define what aspect is sustainable (materials, production, etc.)",
                         "Provide measurable targets"],
        "similar_phrases": ["sustainably made", "sustainability-focused", "eco-sustainable", "green sustainable"],
        "category": "Vague Claims"
    },

    # Carbon/climate terms
    "carbon-neutral": {
        "examples": ["Shell's 'Drive Carbon Neutral'", "Airlines' carbon neutral claims"],
        "alternatives": ["Provide specific emission reduction figures", "Clarify if this includes offsets"],
        "similar_phrases": ["net-zero", "climate neutral", "zero-emissions", "carbon free", "carbon negative"],
        "category": "Carbon Claims"
    },
    "net-zero": {
        "examples": ["Oil companies' net-zero pledges", "Corporate net-zero targets"],
        "alternatives": ["Provide detailed roadmap with interim targets", "Specify scope (1, 2, and 3 emissions)"],
        "similar_phrases": ["carbon neutral", "zero carbon", "climate positive", "carbon balanced"],
        "category": "Carbon Claims"
    },

    # Nature-related terms
    "natural": {
        "examples": ["'Natural' gas marketing", "'All-natural' product claims"],
        "alternatives": ["Specify natural ingredients percentage", "Clarify processing methods"],
        "similar_phrases": ["all-natural", "naturally derived", "nature-based", "plant-based"],
        "category": "Natural Claims"
    },
    "biodegradable": {
        "examples": ["'Biodegradable' plastics that require industrial facilities",
                     "Products labeled biodegradable without timeframe"],
        "alternatives": ["Specify degradation timeframe and conditions", "Provide certification details"],
        "similar_phrases": ["compostable", "naturally decomposing", "eco-degradable", "plant-degradable"],
        "category": "Natural Claims"
    },

    # Recycling terms
    "recyclable": {
        "examples": ["Products labeled recyclable where facilities don't exist", "'Recyclable' single-use plastics"],
        "alternatives": ["Specify recycling rates in target markets",
                         "Clarify 'technically recyclable' vs 'actually recycled'"],
        "similar_phrases": ["recycled content", "post-consumer recycled", "upcycled", "circular"],
        "category": "Recycling Claims"
    },

    # Energy terms
    "renewable energy": {
        "examples": ["Companies using minimal renewables but making broad claims",
                     "Energy providers with small green offerings"],
        "alternatives": ["Specify percentage of renewable energy used", "Provide timeline for full transition"],
        "similar_phrases": ["clean energy", "green power", "sustainable energy", "low-carbon energy"],
        "category": "Energy Claims"
    },

    # New additions
    "regenerative": {
        "examples": ["Food brands using regenerative without certification",
                     "Fashion brands with minimal regenerative practices"],
        "alternatives": ["Provide certification details", "Specify acreage/percentage using regenerative practices"],
        "similar_phrases": ["restorative", "climate-smart", "soil-positive", "carbon farming"],
        "category": "Agricultural Claims"
    },
    "ocean plastic": {
        "examples": ["Products using minimal ocean plastic with prominent labeling",
                     "Claims of 'saving' oceans while still producing plastic"],
        "alternatives": ["Specify percentage of ocean plastic used", "Provide collection verification"],
        "similar_phrases": ["marine plastic", "sea waste", "ocean-bound plastic", "recovered shoreline plastic"],
        "category": "Plastic Claims"
    }
}


# Generate embeddings for all terms and similar phrases
@st.cache_data
def get_term_embeddings():
    all_phrases = []
    phrase_to_term = {}

    for term, data in greenwashing_terms.items():
        all_phrases.append(term)
        phrase_to_term[term] = term
        for phrase in data.get("similar_phrases", []):
            all_phrases.append(phrase)
            phrase_to_term[phrase] = term

    embeddings = model.encode(all_phrases, convert_to_tensor=True)
    return {phrase: embeddings[i] for i, phrase in enumerate(all_phrases)}, phrase_to_term


term_embeddings, phrase_to_term = get_term_embeddings()


def find_similar_terms(text, threshold=0.6):
    """Find similar terms using semantic similarity."""
    text_embedding = model.encode(text, convert_to_tensor=True)
    similarities = {}

    for phrase, embedding in term_embeddings.items():
        sim = util.pytorch_cos_sim(text_embedding, embedding).item()
        if sim > threshold:
            similarities[phrase] = sim

    # Group by original term
    results = {}
    for phrase, sim in similarities.items():
        term = phrase_to_term[phrase]
        if term not in results or sim > results[term]["similarity"]:
            results[term] = {
                "matched_phrase": phrase,
                "similarity": sim,
                "data": greenwashing_terms[term]
            }

    return results


def check_greenwashing(text):
    """Check for potentially misleading environmental claims including spelling variations."""
    findings = []
    report_data = []

    # Check for exact matches and spelling variations
    words = re.findall(r'\b\w+\b', text.lower())
    for word in words:
        for term, data in greenwashing_terms.items():
            # Exact match
            if word == term.lower():
                findings.append(word)
                report_data.append({
                    'original': word,
                    'term': term,
                    'similarity': 1.0,
                    'data': data
                })
            # Spelling variation (edit distance <= 2 for words > 3 chars)
            elif len(word) > 3 and levenshtein_distance(word, term.lower()) <= 2:
                findings.append(word)
                report_data.append({
                    'original': word,
                    'term': term,
                    'similarity': 0.8,  # Default similarity for spelling variations
                    'data': data
                })

    # Check for similar terms using embeddings
    similar_terms = find_similar_terms(text)
    for term, match in similar_terms.items():
        if not any(f.lower() == match['matched_phrase'].lower() for f in findings):
            findings.append(match['matched_phrase'])
            report_data.append({
                'original': match['matched_phrase'],
                'term': term,
                'similarity': match['similarity'],
                'data': match['data']
            })

    if findings:
        # Generate downloadable report
        report = f"Greenwashing Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        report += "Potential Greenwashing Terms Found:\n"
        for item in report_data:
            if item['original'].lower() == item['term'].lower():
                report += f"- Found: '{item['original']}' (exact match)\n"
            else:
                report += f"- Found: '{item['original']}' → similar to '{item['term']}' ({item['similarity']:.0%})\n"

        # Add examples and suggestions
        examples = set()
        suggestions = set()
        for item in report_data:
            examples.update(item['data']['examples'][:1])
            suggestions.update(item['data']['alternatives'])

        report += "\nExamples of problematic usage:\n"
        for example in examples:
            report += f"- {example}\n"

        report += "\nSuggested improvements:\n"
        for suggestion in suggestions:
            report += f"- {suggestion}\n"

        # Add download button
        st.download_button(
            "📄 Download Report",
            data=report,
            file_name="greenwashing_report.txt",
            mime="text/plain"
        )

        # Simple output showing user's exact terms
        st.markdown(f"""
        <div class='warning-box'>
            <h3 style='color: var(--warning);'>⚠️ Potential Greenwashing Detected</h3>
            <p>The following terms may be misleading: <strong>{', '.join(f'"{f}"' for f in findings)}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Show examples
        st.markdown("<h4>Example cases:</h4>", unsafe_allow_html=True)
        for example in list(examples)[:3]:
            st.markdown(f"""
            <div class='example-box'>
                <p>{example}</p>
            </div>
            """, unsafe_allow_html=True)

        # Show suggestions
        st.markdown("<h4>Suggested improvements:</h4>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='suggestion-box'>
            <ul>
                {''.join([f'<li>{suggestion}</li>' for suggestion in suggestions])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

        return ""

    # If no issues found
    st.markdown("""
    <div class='success-box'>
        <h3 style='color: var(--primary);'>✅ Claim Appears Substantiated</h3>
        <p>Your marketing claim doesn't contain obvious greenwashing terms.</p>
    </div>
    """, unsafe_allow_html=True)

    return ""


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


# Navigation bar (tab-like behavior)
selected = option_menu(
    menu_title=None,
    options=["Greenwashing Checker", "Case Studies"],
    icons=["search", "book"],
    orientation="horizontal",
    styles={
        "nav-link": {
            "text-align": "center",
            "--hover-color": "#43A047",
            "font-size": "16px",
            "padding": "10px",
        },
        "container": {
            "background-color": "#1E1E1E",
            "padding": "10px",
            "border-radius": "8px",
        },
        "icon": {
            "color": "white",
            "font-size": "18px"
        },
        "nav-link-selected": {
            "background-color": "#2E7D32",
            "color": "white"
        }
    }
)

# Main Dashboard Layout
st.title("🌿 Sustainable Marketing Compliance Dashboard")

# Sidebar for country selection with flag icon
st.sidebar.header("🌍 Select Your Market")
selected_country = st.sidebar.selectbox("Country/Region", list(compliance_data.keys()),
                                        format_func=lambda x: f"{compliance_data[x]['icon']} {x}")
st.sidebar.markdown(f"""
<div class='sidebar-info'>
    <h4 style='color: var(--primary);'>Compliance Guidelines for {selected_country}:</h4>
    <p>{compliance_data[selected_country]['guideline']}</p>
    <a href="{compliance_data[selected_country]['link']}" target="_blank">Read official guidelines →</a>
</div>
""", unsafe_allow_html=True)

# Show content based on selected tab
if selected == "Greenwashing Checker":
    st.header("Greenwashing Checker")
    user_text = st.text_area("Enter your marketing claim or slogan:", height=150,
                             placeholder="Example: Our product is 100% eco-friendly and sustainable...")

    # In your analysis button click handler:
    if st.button("Analyze for Greenwashing", type="primary"):
        if not user_text or not user_text.strip():
            st.warning("Please enter some text to analyze")
        else:
            st.markdown("### Analysis Results", unsafe_allow_html=True)
            check_greenwashing(user_text)  # This now handles the display internally

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

elif selected == "Case Studies":
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
