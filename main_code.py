import streamlit as st
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
from datetime import datetime
from collections import defaultdict

# Set page config must be first
st.set_page_config(layout="wide")


# Then load other components
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


model = load_model()


# Enhanced Levenshtein distance function for spelling variations
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    # Convert to lowercase for case-insensitive comparison
    s1 = s1.lower()
    s2 = s2.lower()

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


# Function to check for spelling variations
def is_spelling_variation(word, term, max_distance=2):
    # Check if word is a substring of term or vice versa
    if word.lower() in term.lower() or term.lower() in word.lower():
        return True

    # Check Levenshtein distance for short words (length <=5)
    if len(word) <= 5 or len(term) <= 5:
        return levenshtein_distance(word, term) <= 1

    # For longer words, allow slightly more variation
    return levenshtein_distance(word, term) <= max_distance


# Navigation system
def navigation():
    # Create three columns for our tabs
    col1, col2, col3 = st.columns(3)

    # Initialize query params if not present
    if 'tab' not in st.query_params:
        st.query_params['tab'] = 'checker'

    # First tab - Greenwashing Checker
    with col1:
        if st.button("🔍 Greenwashing Checker",
                     use_container_width=True,
                     type="primary" if st.query_params.tab == "checker" else "secondary"):
            st.query_params['tab'] = "checker"
            st.rerun()

    # Second tab - Case Studies
    with col2:
        if st.button("📚 Case Studies",
                     use_container_width=True,
                     type="primary" if st.query_params.tab == "cases" else "secondary"):
            st.query_params['tab'] = "cases"
            st.rerun()

    # Third tab - About
    with col3:
        if st.button("ℹ️ About",
                     use_container_width=True,
                     type="primary" if st.query_params.tab == "about" else "secondary"):
            st.query_params['tab'] = "about"
            st.rerun()

    # Return the current tab
    return st.query_params.tab

# Custom CSS with dark mode-inspired color scheme
st.markdown(
    """
    <style>
    :root {
        --primary: #4CAF50;
        --secondary: #2E7D32;
        --accent: #2E7D32;
        --light-bg: #FAF9F6;
        --card-bg: #FFFFFF;
        --text-light: #000000;
        --text-muted: #555555;
        --warning: #FF9800;
        --danger: #F44336;
    }

    body {
        background-color: var(--light-bg);
        color: var(--text-light);
        font-family: 'Helvetica Neue', 'Segoe UI', sans-serif;
        line-height: 1.6;
    }

    .stApp {
        background-color: var(--light-bg);
    }

    .sidebar .sidebar-content {
        background-color: #F2F1ED;
        border-right: 1px solid #DDD;
    }

    /* SELECTBOX STYLING */
    .stSelectbox > div > div > select {
        background-color: "#FFFFFF" !important;
        color: "#FFFFFF" !important;
        border: 3px solid var(--secondary) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-weight: 500;
    }

    .stSelectbox > div > div > svg {
        color: "#FFFFFF" !important;
    }

    .stSelectbox > div > div > div > div {
        background-color: "#FFFFFF" !important;
        color: "#FFFFFF" !important;
    }

    .stSelectbox > div > div > div > div:hover {
        background-color: #E8F5E9 !important;
    }

    .stTextArea, .stTextInput {
        background-color: var(--card-bg);
        color: var(--text-light);
        border-radius: 10px;
        border: 1px solid #CCC;
        padding: 12px;
    }

    .stButton>button {
        background-color: var(--primary);
        color: #FFF;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-size: 16px;
    }

    .stButton>button:hover {
        background-color: #103611;
    }

    .stMarkdown {
        font-size: 16px;
        color: var(--text-light);
    }

    .stExpander {
        background-color: var(--card-bg);
        border-radius: 10px;
        border: 1px solid #DDD;
        margin-bottom: 16px;
    }

    .warning-box {
        background-color: #FFF3E0;
        border-left: 5px solid var(--warning);
        padding: 16px;
        border-radius: 6px;
        margin-bottom: 16px;
        color: var(--text-light);
    }

    .success-box {
        background-color: #E8F5E9;
        border-left: 5px solid var(--primary);
        padding: 16px;
        border-radius: 6px;
        margin-bottom: 16px;
        color: var(--text-light);
    }

    .example-box {
        background-color: #F1F8E9;
        border-left: 5px solid var(--secondary);
        padding: 14px;
        border-radius: 6px;
        margin: 14px 0;
        color: var(--text-light);
    }

    .term-highlight {
        background-color: var(--accent);
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
        color: #FFFFFF;
    }

    .sidebar-info {
        background-color: #FAF9F6;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #DDD;
        margin-top: 15px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--primary);
    }

    a {
        color: var(--secondary);
        text-decoration: none;
    }

    .footer {
        text-align: center;
        margin-top: 30px;
        padding: 10px;
        color: var(--text-muted);
        font-size: 14px;
    }

    .methodology-box {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid var(--accent);
        margin-bottom: 20px;
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
    term_variations = defaultdict(list)

    # First collect all base terms and their similar phrases
    for term, data in greenwashing_terms.items():
        all_phrases.append(term)
        phrase_to_term[term] = term
        term_variations[term].append(term)

        for phrase in data.get("similar_phrases", []):
            all_phrases.append(phrase)
            phrase_to_term[phrase] = term
            term_variations[term].append(phrase)

    # Generate embeddings for all collected phrases
    embeddings = model.encode(all_phrases, convert_to_tensor=True)

    # Create embedding dictionary
    embeddings_dict = {phrase: embeddings[i] for i, phrase in enumerate(all_phrases)}

    return embeddings_dict, phrase_to_term, term_variations


term_embeddings, phrase_to_term, term_variations = get_term_embeddings()


def find_similar_terms(text, threshold=0.6):
    """Find similar terms using semantic similarity and spelling variations."""
    text_embedding = model.encode(text, convert_to_tensor=True)
    similarities = {}

    # First check for exact matches and spelling variations in all terms and similar phrases
    words = re.findall(r'\b\w+\b', text.lower())
    for word in words:
        for term, variations in term_variations.items():
            for variation in variations:
                if is_spelling_variation(word, variation.lower()):
                    # Calculate similarity score based on edit distance
                    distance = levenshtein_distance(word, variation.lower())
                    similarity = max(0, 1 - (distance / max(len(word), len(variation))))
                    if similarity > threshold:
                        similarities[variation] = similarity

    # Then check semantic similarity
    for phrase, embedding in term_embeddings.items():
        sim = util.pytorch_cos_sim(text_embedding, embedding).item()
        if sim > threshold:
            # Only add if we don't already have a better match
            if phrase not in similarities or sim > similarities[phrase]:
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

    # First check for exact matches and spelling variations in all terms and similar phrases
    similar_terms = find_similar_terms(text)

    for term, match in similar_terms.items():
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


# Main Dashboard Layout
st.title("🌿 Sustainable Marketing Compliance Dashboard")
selected = navigation()

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

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
if selected == "checker":
    st.header("Greenwashing Checker")
    user_text = st.text_area("Enter your marketing claim or slogan:", height=150,
                             placeholder="Example: Our product is 100% eco-friendly and sustainable...")

    if st.button("Analyze for Greenwashing", type="primary"):
        if not user_text or not user_text.strip():
            st.warning("Please enter some text to analyze")
        else:
            st.markdown("### Analysis Results", unsafe_allow_html=True)
            check_greenwashing(user_text)

elif selected == "cases":
    st.header("Green Marketing Case Studies")

    with st.expander("**✅ Positive Examples**", expanded=True):
        st.markdown("""
        <div class='success-box'>
            <h3>Patagonia's Environmental Initiatives</h3>
            <p><strong>Why it works:</strong> Specific, measurable claims with full transparency</p>
            <ul>
                <li>100% of our cotton is organic - verifiable claim</li>
                <li>1% of sales donated to environmental causes - concrete commitment</li>
                <li>Worn Wear program for used gear - circular economy in action</li>
                <p><a href="https://initiatives.weforum.org/industry-net-zero-accelerator/case-study-details/patagonia---sparking-the-sustainability-cultural-shift-at-every-level/aJYTG00000001H34AI" target="_blank">Read More...</a></p>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='success-box'>
            <h3>McCormick & Company's Sustainable Incentives</h3>
            <p><strong>Why it works:</strong> Clear labeling with scientific backing</p>
            <ul>
                <li>The company has SBTi validated targets to reach net zero by 2050</li>
                <li>Publicly reports progress against these targets in their annual Purpose-led Performance Report, ensuring transparency and accountability.</li>
                <li>Collaborates with third-party organizations for independent assessment and verification of their sustainability efforts.</li>
                <p><a href="https://www.packworld.com/sustainable-packaging/article/22921744/mccormicks-michael-okoroafor-discusses-scope-3-emissions">Read More...</a></p>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("**⚠️ Controversial Examples**", expanded=True):
        st.markdown("""
        <div class='warning-box'>
            <h3>Volkswagen's 'Clean Diesel' Campaign</h3>
            <p><strong>Why it failed:</strong> Deliberate deception with technical claims</p>
            <ul>
                <li>Claimed low emissions while using defeat devices</li>
                <li>Resulted in complete campaign withdrawal</li>
                <p><a href="https://www.vox.com/2015/9/21/9365667/volkswagen-clean-diesel-recall-passenger-cars" target="_blank">Read More...</a></p>
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
                <p><a href="https://trellis.net/article/fiji-water-sued-over-claim-product-carbon-negative/" target="_blank">Read More...</a></p>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif selected == "about":
    st.header("About This Tool")

    st.markdown("""
    <div class='methodology-box'>
        <h3>Purpose</h3>
        <p>This tool helps businesses and sustainability professionals identify potential greenwashing in their communications by:</p>
        <ul>
            <li>Detecting vague or misleading environmental claims</li>
            <li>Providing alternative phrasing suggestions</li>
            <li>Offering country-specific compliance guidelines</li>
            <li>Showing real-world examples of both good and problematic marketing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='methodology-box'>
            <h3>Features</h3>
            <p>The tool uses multiple techniques to identify potential greenwashing:</p>
            <ul>
                <li> Country Specific Guidelines: Regulatory requirements from major markets to ensure regional compliance</li>
                <li> Term Matching: Exact matching against a database of known problematic terms (e.g., "eco-friendly", "sustainable")</li>
                <li> Limitations & Examples: The tool points out examples of the user-inputted greenwashed terms and also presents its limitations along with suggestions on how to fix them</li>
                <li> Spelling Variation Detection: Levenshtein distance algorithm to catch misspellings (e.g., "eco-frendly")</li>
                <li> Semantic Similarity: SentenceTransformer embeddings (all-MiniLM-L6-v2 model) to detect conceptually similar phrases</li>
                <li> Hosted on Streamlit using GitHub: The code is public, allowing any business to fork and modify the tool to align with their own sustainability or compliance standards</li>
                <li> Case Studies: Real-world examples have been included to provide context.</li>
                <li> Report Download: Users can download a detailed greenwashing report summarizing the flagged terms, context, and suggestions for revision</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='methodology-box'>
            <h3>Methodology</h3>
            <p>This tool aims to detect potential greenwashing in marketing content. To gather relevant and effective resources, an extensive search strategy was implemented that utilized a range of research materials provided by WonderWorks, country-specific regulatory guidelines, generative AI tools, and both academic and popular sources.</p>
            <p>Sources and Databases</p>
            <ul>
                <li>WonderWorks research material: Their initial documents and shared links offered a foundational understanding of greenwashing practices, supplying key datasets and case studies that guided the research direction.</li>
                <li>Country-specific regulatory guidelines: Used to ensure compliance with major regional frameworks governing environmental marketing claims.</li>
                <li>Google Scholar: This tool was used to locate a wide range of academic literature, including peer-reviewed articles and case studies on sustainability and marketing ethics. It also played a key role in developing the greenwashing checker by helping identify commonly used greenwashing terms.</li>
                <li>Google Search: Used to find popular sources, including blog posts, news articles, and government or NGO reports, to understand current trends and public perspectives on greenwashing.</li>
            </ul>
            <p></p>
            <p>Keywords:</p>
            <ul>
                <li>Greenwashing</li>
                <li>Environmental marketing claims</li>
                <li>Sustainability standards</li>
                <li>ESG compliance</li>
                <li>Eco-labeling and certification</li>
                <li>Advertising regulation</li>
                <li>Regulatory enforcement</li>
                <li>Green claim substantiation</li>
            </ul>
            <p>These keywords, along with associated terms, helped in capturing a wide array of sources that addressed different aspects of environmental marketing and greenwashing detection.</p>
            <p></p>
            <p>Search Strategies</p>
            <ul>
                <li>Filtering by relevance: Search results were sorted to prioritize the most directly related sources.</li>
                <li>Filtering by publication year: Only materials published after 2010 were included to reflect the recent evolution of green marketing and ESG policies.</li>
                <li>Use of Generative AI tools: ChatGPT and DeepSeek were utilized to explore and refine search strategies, generate relevant keyword variations, and uncover less obvious greenwashing tactics.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='methodology-box'>
        <h3>Limitations</h3>
        <ul>
            <li>This tool identifies <strong>potential</strong> issues but cannot assess claim validity</li>
            <li>Does not verify scientific accuracy of environmental claims</li>
            <li>Should be used as a screening tool, not a definitive compliance assessment</li>
            <li>While the tool can detect spelling mistakes and semantic similarities, it still struggles to do so accurately and reliably in many cases.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='methodology-box'>
        <h3>Acknowledgements</h3>
        <p>This tool was developed as part of the Future17 Program</p>
        <ul>
            <li>Designed and prepared for <a href="https://www.wonderwork.digital/" target="_blank">WonderWork</a>.</li>
            <li>This tool was designed and coded by Team 09 - Group 1 | Future17 2025</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Made for the Future17 Program 2025
</div>
""", unsafe_allow_html=True)
