from __future__ import annotations


"""
Prompt templates aligned to FERMI appendix where available.

Sources:
- OPRO optimization prompt: FERMI Appendix Figure 9
- FERMI optimization prompt: FERMI Appendix Figure 8
- Few-shot template: FERMI Appendix Listing 3
- Vanilla (LaMP_rate): FERMI Appendix Listing 6

Notes:
- LaMP_tag / LaMP_title vanilla templates are minimal adjustments because the
  appendix only explicitly provides vanilla text for LaMP_rate (Listing 6).
"""


# Appendix Listing 6 (verbatim structure, generalized placeholders).
VANILLA_LAMP_RATE_LISTING6 = (
    "Answer to the given question. Just answer with 1, 2, 3, 4, or 5 without further explanation:\n"
    "Question:{question}\n"
    "Answer choices:{answer_choices}\n"
    "Answer:"
)


# Minimal adjustment from Listing 6 to LaMP_tag classification.
VANILLA_LAMP_TAG_MIN_ADJUST = (
    "Choose the proper answer to the given question among the given answer choices. "
    "Your answer should be a single label among given answer choices:\n"
    "Question:{question}\n"
    "Answer choices:{answer_choices}\n"
    "Answer:"
)


# Minimal adjustment from Listing 6 to LaMP_title generation.
VANILLA_LAMP_TITLE_MIN_ADJUST = (
    "Generate a title for the following abstract. Return only one concise title without further explanation.\n"
    "Question:{question}\n"
    "Answer choices:{answer_choices}\n"
    "Answer:"
)


# Appendix Listing 3 (few-shot format; placeholders are directly fillable).
FEWSHOT_LISTING3 = (
    "{retrieved_block}\n"
    "Based on the above previous questions and answers, choose the proper answer to the given question among the given answer choices.\n"
    "Question:{question}\n"
    "Answer choices:{answer_choices}\n"
    "Answer:"
)


# Appendix Figure 9 (OPRO p_opt template structure).
OPRO_FIGURE9_POPT = (
    "I have some texts along with their corresponding scores. "
    "The texts are arranged in ascending order based on their scores, where higher scores indicate better quality.\n"
    "{memory_block}\n"
    "The following exemplars show how to apply your text: you replace <INS> in each input with your text, "
    "then read the input and give an output. We say your output is wrong if your output is different from the given output, "
    "and we say your output is correct if they are the same.\n"
    "{demonstration_block}\n"
    "Write your new text that is different from the old ones and has a score as high as possible. "
    "Write the text in square brackets."
)


# Appendix Figure 8 (FERMI p_opt template structure).
FERMI_FIGURE8_POPT = (
    "I want to find the text that could make you a personalized answer for the task ({task_focus}), "
    "based on the given personal information.\n"
    "For {num_questions} questions, I have the responses from this person and you. "
    "If your response is identical to the person's, then you get a higher score. "
    "Here, I have some previous texts along with their corresponding average scores and specific cases "
    "that you failed to correctly answer. The texts are arranged in ascending order based on their scores, "
    "where higher scores indicate better quality.\n"
    "{memory_block}\n"
    "The following exemplars show how to apply your text: you replace <INS> in each input with your text, "
    "then read the input and give an output.\n"
    "{demonstration_block}\n"
    "Write your new text that is different from the old ones and has a score as high as possible. "
    "Write the text in square brackets."
)


DEFAULT_TEMPLATES = {
    "LaMP2_tag": VANILLA_LAMP_TAG_MIN_ADJUST,
    "LaMP3_rate": VANILLA_LAMP_RATE_LISTING6,
    "LaMP5_title": VANILLA_LAMP_TITLE_MIN_ADJUST,
}
