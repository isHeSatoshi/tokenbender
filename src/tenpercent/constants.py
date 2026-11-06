SEED = 20261106
MODEL_ID = "johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent"
MODEL_REVISION = "6c13ee7e6a4d229615dccdd089869818ad389388"
MODEL_SHA256 = "2084675316db424191877b0ee41689ba28b83e07c8444fc1f44f037c769e82dc"
BASE_PARAMETERS = 1_543_714_304
BASE_BYTES = 3_087_467_144
PARAMETER_CAP = BASE_PARAMETERS // 10
BYTE_CAP = BASE_BYTES // 10
SOURCE_TARGET = 152_800_000
BANKING_ID = "PolyAI/banking77"
BANKING_REVISION = "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"
CLINC_ID = "clinc/clinc_oos"
CLINC_REVISION = "155b9c710419136e17307b80d0a13e68cd46b4ec"
SYSTEM_PROMPT = "Classify the banking customer message into its intent label."
NEW_INTENTS = (
    "credit_score",
    "improve_credit_score",
    "apr",
    "routing",
    "direct_deposit",
    "interest_rate",
    "credit_limit",
    "credit_limit_change",
    "min_payment",
    "pay_bill",
    "rewards_balance",
    "redeem_rewards",
)
BODY_SUFFIXES = tuple(
    f"{module}.{projection}.weight"
    for module, projections in (
        ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
        ("mlp", ("gate_proj", "up_proj", "down_proj")),
    )
    for projection in projections
)
