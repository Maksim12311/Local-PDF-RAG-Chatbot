from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)

from deepeval.models import OllamaModel


# -----------------------------------
# Local evaluator model
# -----------------------------------

evaluation_model = OllamaModel(
    model="llama3.1",
    base_url="http://localhost:11434",
    temperature=0
)


# -----------------------------------
# Example RAG results
# -----------------------------------

test_cases = [
    LLMTestCase(
        input="Who is the author?",
        actual_output="The author is Maksim Brusilovskii.",
        retrieval_context=[
            "The author of Smart Information Systems Engineering Assignment 1 is Maksim Brusilovskii."
        ]
    ),

    LLMTestCase(
    input="What happens if the football field is occupied?",
    actual_output="The players have to wait because another group is still using the field.",
    retrieval_context=[
        "Another group is still using the field, so the players have to wait before the training can start."
    ]
),

    LLMTestCase(
        input="What is the problem with food delivery?",
        actual_output="The courier cannot find the flat.",
        retrieval_context=[
            "The problem with the food delivery is that the courier cannot find the flat."
        ]
    ),
]


# -----------------------------------
# Metrics
# -----------------------------------

answer_relevancy = AnswerRelevancyMetric(
    threshold=0.5,
    model=evaluation_model,
    include_reason=True
)

faithfulness = FaithfulnessMetric(
    threshold=0.5,
    model=evaluation_model,
    include_reason=True
)

context_relevancy = ContextualRelevancyMetric(
    threshold=0.5,
    model=evaluation_model,
    include_reason=True
)


# -----------------------------------
# Run evaluation
# -----------------------------------

evaluate(
    test_cases=test_cases,
    metrics=[
        answer_relevancy,
        faithfulness,
        context_relevancy
    ]
)