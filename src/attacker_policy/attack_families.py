"""Mapping from attacker actions to the prompt-injection families they cover."""

ATTACK_CATEGORY_FAMILIES = {
    "direct_override": {"direct_injection"},
    "data_exfiltration": {"secret_extraction"},
    "tool_abuse": {"tool_confusion", "unauthorized_operation"},
    "destructive_action": {"unauthorized_operation"},
    "authority_impersonation": {"indirect_injection"},
    "fictional_roleplay": {"novel_evasion"},
    "obfuscated_encoding": {"novel_evasion"},
    "delimiter_smuggling": {"novel_evasion"},
    "logical_constraint_framing": {"novel_evasion"},
    "multilingual_injection": {"novel_evasion"},
}

DEFAULT_ATTACK_CATEGORIES = (
    "direct_override", "data_exfiltration", "tool_abuse",
    "destructive_action", "authority_impersonation",
)
