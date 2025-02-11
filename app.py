# code with better interface
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import streamlit as st
from PIL import Image, ImageDraw

# Load BERT model and tokenizer
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

tokenizer, model = load_bert_model()

# Define economic concept categories with a real thesaurus approach
economic_thesaurus = {
    "demand": ["buyers", "consumers", "market demand", "purchase curve", "customer need", "demand"],
    "supply": ["producers", "sellers", "market supply", "production curve", "goods available", "supply"],
    "slope": ["rate of change", "gradient", "elasticity", "steepness", "rate", "slope", "decreases at", "increases at"],
    "intercept": ["starting value", "y-intercept", "fixed amount", "base level", "intercept", "initial", "has an intercept of"],
    "increase": ["rises by", "grows at a rate of", "climbs", "expands", "goes up", "increase"],
    "decrease": ["drops by", "shrinks at a rate of", "declines", "contracts", "goes down", "decrease"]
}

def get_bert_embedding(text):
    """
    Generates a BERT embedding for a given text.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=10)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Extract CLS token representation

def find_best_match(word_or_phrase):
    """
    Uses BERT similarity to find the closest match in the economic thesaurus.
    """
    input_vector = get_bert_embedding(word_or_phrase).reshape(1, -1)
    best_match = None
    highest_similarity = 0.0

    for category, synonyms in economic_thesaurus.items():
        for synonym in synonyms:
            synonym_vector = get_bert_embedding(synonym).reshape(1, -1)
            similarity = cosine_similarity(input_vector, synonym_vector)[0][0]

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = category

    return best_match if highest_similarity > 0.7 else None  # 0.7 threshold for strong similarity

def extract_equation_params_thesaurus(text):
    """
    Extracts demand and supply equation parameters using NLP.
    """
    sentences = re.split(r'(?<=\.)\s', text.lower())
    numbers = [float(num) for num in re.findall(r'\d+\.*\d*', text)]
    st.write(f"🔹 Extracted Numbers: {numbers}")

    if len(numbers) < 4:
        return "❌ Error: Less than 4 numbers extracted."

    demand_values = {"slope": None, "intercept": None}
    supply_values = {"slope": None, "intercept": None}
    detected_demand = False
    detected_supply = False

    for sentence in sentences:
        words = sentence.split()
        matched_terms = [find_best_match(word) for word in words]
        matched_terms = list(filter(None, matched_terms))

        is_demand = "demand" in matched_terms and not detected_demand
        is_supply = "supply" in matched_terms and not detected_supply

        if is_demand:
            st.write(f"✅ Detected Demand Phrase: {sentence.strip()}")
            detected_demand = True
            sentence_numbers = [float(num) for num in re.findall(r'\d+\.*\d*', sentence)]
            if len(sentence_numbers) >= 2:
                demand_values["intercept"], demand_values["slope"] = sentence_numbers[:2]

        if is_supply:
            st.write(f"✅ Detected Supply Phrase: {sentence.strip()}")
            detected_supply = True
            sentence_numbers = [float(num) for num in re.findall(r'\d+\.*\d*', sentence)]
            if len(sentence_numbers) >= 2:
                supply_values["intercept"], supply_values["slope"] = sentence_numbers[:2]

    if None in demand_values.values() or None in supply_values.values():
        return "❌ Error: Could not determine function parameters correctly."

    demand_values["slope"] = -abs(demand_values["slope"])

    result = (
        f"✅ FINAL EXTRACTED VALUES:\n"
        f"Demand Intercept: {demand_values['intercept']}\n"
        f"Demand Slope: {demand_values['slope']}\n"
        f"Supply Intercept: {supply_values['intercept']}\n"
        f"Supply Slope: {supply_values['slope']}"
    )
    return result

def create_placeholder_image(text, filename):
    """ Creates a simple placeholder image with text. """
    img = Image.new('RGB', (100, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((20, 40), text, fill="black")
    img.save(filename)
    print(f"✅ Placeholder image created as '{filename}'")

create_placeholder_image("Demand", "demand_icon.png")
create_placeholder_image("Supply", "supply_icon.png")

def main():
    st.title("📚 Economics Equation Extractor")
    st.markdown("""
    Welcome to the **Economics Equation Extractor**!  
    This app helps you extract demand and supply equation parameters from text using NLP.
    """)

    demand_icon = Image.open("demand_icon.png")
    supply_icon = Image.open("supply_icon.png")

    st.sidebar.image(demand_icon, caption="Demand", width=100)
    st.sidebar.image(supply_icon, caption="Supply", width=100)

    text_input = st.text_area("Enter Economics Text:", height=200, placeholder="Paste your text here...")

    if st.button("Extract Equation Parameters"):
        if not text_input.strip():
            st.error("Please enter some text.")
        else:
            result = extract_equation_params_thesaurus(text_input)
            st.success(result)

    st.markdown("### Example Input:")
    example_text = """
    The demand function has an intercept of 10 and decreases at a rate of 4 per unit. 
    The supply function has an intercept of 2 and a slope of 5.
    """
    st.code(example_text)

if __name__ == "__main__":
    main()
