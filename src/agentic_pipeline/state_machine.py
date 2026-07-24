from typing import Dict, Any, List, Optional
from src.agentic_pipeline.mcp_tools import MCPRepositoryTools
from src.agentic_pipeline.agents.investigator import InvestigatorAgent
from src.agentic_pipeline.agents.developer import DeveloperAgent
from src.agentic_pipeline.agents.validator import ValidatorAgent

class MultiAgentRemediationPipeline:
    """
    LangGraph Multi-Agent Orchestration State Machine.
    Coordinates Investigator, Developer, and Validator agents to autonomously 
    diagnose, patch, and verify breaking API data pipeline anomalies.
    """

    def __init__(self, workspace_root: Optional[str] = None):
        self.mcp_tools = MCPRepositoryTools(workspace_root=workspace_root)
        self.investigator = InvestigatorAgent(self.mcp_tools)
        self.developer = DeveloperAgent(self.mcp_tools)
        self.validator = ValidatorAgent(self.mcp_tools)

    def run_remediation_workflow(self, anomaly_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the autonomous end-to-end multi-agent pipeline:
        1. Investigator: Diagnoses root cause and broken payload keys.
        2. Developer: Writes backward-compatible python patch code.
        3. Validator: Runs AST parser and test verification suite.
        """
        trace = []
        
        # Step 1: Investigation Phase
        trace.append({"step": "INVESTIGATION", "status": "IN_PROGRESS"})
        diagnosis = self.investigator.investigate(anomaly_event)
        trace.append({"step": "INVESTIGATION", "status": "COMPLETED", "output": diagnosis})

        # Step 2: Patch Development Phase
        trace.append({"step": "DEVELOPMENT", "status": "IN_PROGRESS"})
        patch_solution = self.developer.generate_patch(diagnosis)
        trace.append({"step": "DEVELOPMENT", "status": "COMPLETED", "output": patch_solution})

        # Step 3: Validation & Sandbox Testing Phase
        trace.append({"step": "VALIDATION", "status": "IN_PROGRESS"})
        validation_res = self.validator.validate_patch(patch_solution)
        trace.append({"step": "VALIDATION", "status": "COMPLETED", "output": validation_res})

        # Step 4: Final Synthesis & PR Status
        workflow_success = validation_res.get("validated", False)
        
        return {
            "anomaly_id": anomaly_event.get("id"),
            "success": workflow_success,
            "diagnosis": diagnosis,
            "patch_solution": patch_solution,
            "validation_results": validation_res,
            "execution_trace": trace,
            "status": "REMEDIATED" if workflow_success else "FAILED_VALIDATION"
        }
