"""Edit these for the business case. Prompts ARE the product in a hackathon."""
SYSTEM = "You are an expert assistant for [FILL IN: company/domain]. Be concise and accurate."

TASKS = {
    "Summarize": "Summarize the text below in 5 bullet points for a busy manager.\n\n{text}",
    "Extract (JSON)": "Extract the key entities, dates, amounts and action items from the text. "
                      "Return JSON with keys: entities, dates, amounts, action_items.\n\n{text}",
    "Classify + risk": "Classify the text (category, urgency 1-5, sentiment) and list risks. "
                       "Return JSON with keys: category, urgency, sentiment, risks.\n\n{text}",
    "Recommend next steps": "You are a business analyst. Based on the text, give 3 prioritized "
                            "recommendations with reasoning and expected impact.\n\n{text}",
}

# Similar-cases tab (incidents / tickets / support cases)
CASE_SYSTEM = ("You are a support engineer. Using ONLY the similar past cases provided, "
               "propose how to resolve the new case. Give numbered steps and cite the past "
               "cases like [12]. If the past cases do not fit, say so.")
CASE_JSON = ("New case: {q}\n\nSimilar past cases:\n{ctx}\n\n"
             "Return JSON with keys: category, priority (Low/Medium/High), likely_root_cause, "
             "suggested_steps (list), confidence (0-1), based_on (list of case numbers).")
