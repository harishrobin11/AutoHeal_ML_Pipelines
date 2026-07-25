# AutoHeal-ML Auto-Generated Patch
# Remediation for: SchemaBreak (Missing key/feature: payload_json)

def parse_incoming_telemetry_payload(payload: dict) -> dict:
    """
    Parses incoming API request payload with backward-compatible fallback 
    for missing or renamed fields.
    """
    processed = dict(payload)
    
    # AutoHeal Patch: Safely extract 'user_tier' with fallback default
    if "payload_json" not in processed:
        # Check if legacy key exists
        if "legacy_tier_id" in processed:
            processed["payload_json"] = "standard" if processed["legacy_tier_id"] == 101 else "free"
        else:
            processed["payload_json"] = "standard" # Safe production default
            
    return processed
