"""
Recommendation Agent — generates investment analysis and recommendations.

Takes market data from the Market Agent and the user's profile to produce
a personalised investment thesis explaining the reasoning behind the
recommendation.

TODO: Replace stub with LangChain agent using a specialised LLM.
"""


class RecommendationAgent:
    """
    Generates a structured investment recommendation.

    The recommendation includes:
        - Summary of relevant market conditions
        - Analysis tailored to the user's risk profile and horizon
        - Clear investment thesis with supporting factors
        - Suggested actions
    """

    def __init__(self, model=None):
        self.model = model

    def run(self, query: str, market_data: dict = None, user_profile: dict = None) -> dict:
        """
        Generate an investment recommendation.

        Args:
            query: User's original question.
            market_data: Output from the Market Agent.
            user_profile: Dict with risk_level, horizon, capital, goals.

        Returns:
            Dict with 'action', 'result', and 'response' (the recommendation text).
        """
        # TODO: Implement LLM-based recommendation generation
        profile_desc = ""
        if user_profile:
            profile_desc = (
                f"Perfil: {user_profile.get('risk_level', 'N/A')}, "
                f"horizonte: {user_profile.get('investment_horizon', 'N/A')}"
            )

        return {
            "action": "Generando análisis y recomendación personalizada",
            "result": f"Análisis realizado. {profile_desc}",
            "response": f"Recomendación placeholder para: {query[:100]}",
        }
