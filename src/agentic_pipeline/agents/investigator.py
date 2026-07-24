import json
from typing import Dict, Any
from src.agentic_pipeline.mcp_tools import MCPRepositoryTools

class InvestigatorAgent:
    """
    Investigator Agent in LangGraph Multi-Agent Architecture.
    Analyzes failure anomaly details, SHAP feature attributions, error stack traces,
    and searches codebase files using MCP tools to diagnose root cause.
    """

    def __init__(self, mcp_tools: MCPRepositoryTools):
        self.mcp_tools = mcp_tools

    def investigate(self, anomaly_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes root cause investigation on an anomaly event.
        Returns diagnostic findings and target file locations requiring patch.
        """
        service = anomaly_event.get("service_name", "unknown-service")
        endpoint = anomaly_event.get("endpoint", "")
        root_cause = anomaly_event.get("root_cause_category", "SchemaBreak")
        top_feature = anomaly_event.get("top_feature", "")

        # Search codebase for matching service endpoints or schemas
        search_res = self.mcp_tools.search_codebase_symbol(endpoint)
        if search_res["match_count"] == 0:
            search_res = self.mcp_tools.search_codebase_symbol(service)

        # Formulate root cause diagnosis
        diagnosis = {
            "service_name": service,
            "endpoint": endpoint,
            "root_cause_category": root_cause,
            "top_feature": top_feature,
            "matched_files": [m["file"] for m in search_res["matches"]],
            "analysis": f"Detected {root_cause} on {endpoint}. Primary breaking feature factor: '{top_feature}'. Codebase search found {search_res['match_count']} symbol references.",
            "recommended_action": f"Update schema validation and add backward compatibility fallback for missing payload key '{top_feature}'."
        }
        
        return diagnosis
