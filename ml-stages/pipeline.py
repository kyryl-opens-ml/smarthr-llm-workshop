from dagster import (
    AssetExecutionContext,
    AssetOut,
    asset,
)

# Asset group: data
@asset(group_name="data")
def raw_data(context: AssetExecutionContext):
    return "raw_data"

@asset(group_name="data")
def training_data(context: AssetExecutionContext, raw_data):
    return "training_data"

@asset(group_name="data")
def test_data(context: AssetExecutionContext, raw_data):
    return "test_data"

@asset(group_name="data")
def labeled_data(context: AssetExecutionContext, raw_data):
    return "labeled_data"

@asset(group_name="data")
def production_data(context: AssetExecutionContext):
    return "production_data"

# Asset group: rag
@asset(group_name="rag")
def lancedb_database(context: AssetExecutionContext, training_data):
    return "lancedb_dataset"

@asset(group_name="rag")
def vector_index(context: AssetExecutionContext, lancedb_database):
    return "vector_index"

@asset(group_name="rag")
def fts_index(context: AssetExecutionContext, lancedb_database):
    return "fts_index"

@asset(group_name="rag")
def reranker(context: AssetExecutionContext):
    return "reranker"

@asset(group_name="rag")
def inference_rag(context: AssetExecutionContext, vector_index, fts_index, reranker):
    return "inference_samples"

@asset(group_name="rag")
def evaluation_rag(context: AssetExecutionContext, inference_rag):
    return "evaluation_rag"

# Asset group: finetuning
@asset(group_name="finetuning")
def chat_dataset(context: AssetExecutionContext, training_data):
    return "chat_dataset"

@asset(group_name="finetuning")
def llama1b(context: AssetExecutionContext, chat_dataset):
    return "llama1b"

@asset(group_name="finetuning")
def llama3b(context: AssetExecutionContext, chat_dataset):
    return "llama3b"

@asset(group_name="finetuning")
def llama8b(context: AssetExecutionContext, chat_dataset):
    return "llama8b"

@asset(group_name="finetuning")
def llama70b(context: AssetExecutionContext, chat_dataset):
    return "llama70b"

@asset(group_name="finetuning")
def llama405b(context: AssetExecutionContext, chat_dataset):
    return "llama405b"

@asset(group_name="finetuning")
def evaluation_finetuning(context: AssetExecutionContext, llama1b, llama3b, llama8b, llama70b, llama405b):
    return "evaluation_finetuning"

# Asset group: ai_team_building_agent
@asset(group_name="ai_team_building_agent")
def company_info(context: AssetExecutionContext):
    return "company_info"

@asset(group_name="ai_team_building_agent")
def sourced_papers(context: AssetExecutionContext):
    return "sourced_papers"

@asset(group_name="ai_team_building_agent")
def candidate_table(context: AssetExecutionContext):
    return "candidate_table"

@asset(group_name="ai_team_building_agent")
def email_drafts(context: AssetExecutionContext):
    return "email_drafts"

@asset(group_name="ai_team_building_agent")
def review(context: AssetExecutionContext):
    return "review"
