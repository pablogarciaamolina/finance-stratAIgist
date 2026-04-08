"""
Orchestrator Agent — routes user queries to the appropriate sub-agents.

The Orchestrator decides which agents to invoke based on the user's query,
coordinates the flow (Market → Recommendation → Critic), and assembles
the final response.

TODO: Replace stub with LangChain-based orchestration logic.
"""


class OrchestratorAgent:
    """
    Central coordinator for the multi-agent pipeline.

    Flow:
        1. Receive user query + profile
        2. Determine which sub-agents are needed
        3. Execute sub-agents in sequence
        4. Assemble and return the final response
    """

    def __init__(self, market_agent=None, recommendation_agent=None, critic_agent=None):
        self.market_agent = market_agent
        self.recommendation_agent = recommendation_agent
        self.critic_agent = critic_agent

    def run(self, query: str, user_profile: dict = None) -> dict:
        """
        Execute the multi-agent pipeline.

        Args:
            query: User's investment question.
            user_profile: Dict with risk_level, investment_horizon, capital, goals.

        Returns:
            Dict with 'response', 'agent_trace', and 'metadata'.
        """
        trace = []

        # Step 1: Orchestrator analysis
        trace.append({
            "agent": "Orchestrator",
            "action": "Analyzing query and determining required agents",
            "result": "Pipeline: Market → Recommendation → Critic",
        })

        # Step 2: Market Agent
        market_data = {}
        if self.market_agent:
            market_data = self.market_agent.run(query)
            trace.append({
                "agent": "Market Agent",
                "action": market_data.get("action", ""),
                "result": market_data.get("result", ""),
            })

        # Step 3: Recommendation Agent
        recommendation = {}
        if self.recommendation_agent:
            recommendation = self.recommendation_agent.run(
                query, market_data=market_data, user_profile=user_profile
            )
            trace.append({
                "agent": "Recommendation Agent",
                "action": recommendation.get("action", ""),
                "result": recommendation.get("result", ""),
            })

        # Step 4: Critic Agent
        final_response = recommendation.get("response", "")
        if self.critic_agent:
            critique = self.critic_agent.run(
                query, recommendation=final_response, market_data=market_data
            )
            trace.append({
                "agent": "Critic Agent",
                "action": critique.get("action", ""),
                "result": critique.get("result", ""),
            })
            final_response = critique.get("revised_response", final_response)

        return {
            "response": final_response,
            "agent_trace": trace,
            "metadata": {"pipeline": "orchestrator"},
        }
