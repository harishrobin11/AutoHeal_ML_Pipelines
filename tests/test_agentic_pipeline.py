import pytest
from src.agentic_pipeline.state_machine import MultiAgentRemediationPipeline
from src.agentic_pipeline.guardrails_eval import SafetyGuardrailsAndEval
from src.backend.pr_automator import GitHubPRAutomator

def test_multi_agent_pipeline():
    pipeline = MultiAgentRemediationPipeline()
    mock_anomaly = {
        "id": "anomaly-test-99",
        "service_name": "user-service",
        "endpoint": "/api/v1/user/profile",
        "root_cause_category": "SchemaBreak",
        "top_feature": "user_tier",
        "z_score": 3.85
    }
    
    res = pipeline.run_remediation_workflow(mock_anomaly)
    assert res["status"] in ["REMEDIATED", "FAILED_VALIDATION"]
    assert "diagnosis" in res
    assert "patch_solution" in res

def test_safety_guardrails():
    safe_code = "def fix(): pass"
    unsafe_code = "import os\nos.system('rm -rf /')"
    
    res_safe = SafetyGuardrailsAndEval.evaluate_patch_safety(safe_code)
    res_unsafe = SafetyGuardrailsAndEval.evaluate_patch_safety(unsafe_code)
    
    assert res_safe["guardrail_passed"] is True
    assert res_unsafe["guardrail_passed"] is False

def test_pr_automator():
    mock_remediation = {
        "anomaly_id": "test-123",
        "diagnosis": {"service_name": "user-service", "endpoint": "/api/v1/user/profile", "root_cause_category": "SchemaBreak"},
        "patch_solution": {"patch_code": "def parse(): pass"}
    }
    
    pr = GitHubPRAutomator.create_pull_request(mock_remediation)
    assert pr["success"] is True
    assert "pr_url" in pr
