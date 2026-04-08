"""
Critic Agent — validates and reviews the recommendation.

Checks the Recommendation Agent's output for inconsistencies, missing
information, unconsidered risks, and logical errors.  Produces a
revised (or confirmed) final response.

TODO: Replace stub with LangChain agent using a specialised LLM.
"""


class CriticAgent:
    """
    Reviews and validates the generated recommendation.

    Checks for:
        - Internal consistency of the analysis
        - Missing risk factors
        - Unsupported claims
        - Alignment with user profile constraints
    """

    def __init__(self, model=None):
        self.model = model

    def run(
        self,
        query: str,
        recommendation: str = "",
        market_data: dict = None,
    ) -> dict:
        """
        Review and optionally revise a recommendation.

        Args:
            query: Original user query.
            recommendation: Text from the Recommendation Agent.
            market_data: Market data used to generate the recommendation.

        Returns:
            Dict with 'action', 'result', and 'revised_response'.
        """
        # TODO: Implement LLM-based critique and revision
        return {
            "action": "Validando coherencia y detectando riesgos no considerados",
            "result": "Revisión completada. Sin incoherencias detectadas.",
            "revised_response": recommendation,
        }
