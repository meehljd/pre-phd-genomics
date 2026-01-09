# Multi-Agent Co-Scientist Architecture for Mayo Clinic

## System architecture: GCP-native multi-agent research platform

Your system requires **6 specialized agents** orchestrated via supervisor pattern, deployed on Vertex AI Agent Engine with Cloud Run for compute-intensive tasks. Based on Google's 7-agent co-scientist (validated in AML/liver fibrosis) and NVIDIA's BioNeMo patterns, here's the production architecture.

### Core agent topology

**Supervisor Agent** (ADK on Agent Engine): Routes tasks, manages state, coordinates workflows. Uses Gemini 2.5 Pro for reasoning + forced function calling for deterministic routing.

**Specialized Workers**:
1. **Literature Agent** (domain-specific): PubMed/ArXiv search, RAG synthesis, citation tracking. Uses Vertex AI Search + custom embeddings (MedCPT/PubMedBERT)
2. **Experiment Design Agent** (task-specific): Protocol generation, resource optimization, hypothesis refinement. Gemini 2.0 with reflection loop
3. **Bioinformatics Agent** (domain-specific): Pipeline orchestration (RNA-seq, ChIP-seq, single-cell). Integrates nf-core workflows, BLAST, AlphaFold via MCP
4. **Clinical Review Agent** (domain-specific): EHR analysis, safety assessment, regulatory compliance. FHIR-compliant, HIPAA-ready
5. **Data Analysis Agent** (task-specific): Statistical analysis, visualization, ML model training. Triggers PyTorch jobs on Vertex AI
6. **Hypothesis Agent** (task-specific): Knowledge graph traversal, mechanism generation, novelty assessment

**Communication**: Agent2Agent (A2A) protocol for inter-agent calls, MCP for tool integration, shared state via Firestore with checkpointing.

## GCP service mapping

### Agent orchestration layer
- **Vertex AI Agent Engine**: Primary runtime for all 6 agents. Managed sessions, context, evaluation, tracing. Handles auth, scaling, fault tolerance
- **ADK (Agent Development Kit)**: Framework for building agents. 100 lines Python per agent. Sequential/parallel/loop workflows. Native A2A/MCP support
- **Agent Garden**: Template library. Start with RAG agent + multi-agent examples

**Why Agent Engine vs alternatives**: Fully managed (no K8s ops), built-in memory/sessions, native trace integration, HIPAA-compliant, billing by vCPU-hour (predictable costs)

### Compute layer
- **Cloud Run**: Serverless containers for MCP servers (bioinformatics tools, database connectors). Auto-scales 0→N, 2nd-gen execution for long tasks
- **Vertex AI Training**: PyTorch model training with custom containers. DDP for multi-GPU (V100/A100). Managed Jupyter for exploration
- **Vertex AI Prediction**: TorchServe-compatible serving for genomics models. Auto-scaling endpoints with traffic split for A/B testing

### Data/memory layer
- **Firestore**: Agent state, conversation history, checkpoints for HITL. Document model ideal for nested research workflows
- **Cloud SQL (PostgreSQL + pgvector)**: Long-term memory, structured metadata. pgvector extension for semantic search
- **Vertex AI Vector Search**: High-scale embeddings (millions of papers). Sub-50ms retrieval. Hybrid search (vector + metadata filters)
- **Cloud Storage**: Raw datasets, model artifacts, pipeline outputs. Lifecycle policies for cost optimization
- **Memorystore (Redis)**: Short-term memory cache, session data, rate limiting

### Orchestration/communication
- **Pub/Sub**: Async agent communication, event-driven triggers. Topic per agent + dead-letter queues
- **Eventarc**: Event routing from GCS (new dataset uploaded) → trigger bioinformatics agent
- **Workflows**: Multi-step orchestrations across agents. Visual DAG, error handling, parallel branches. Integrates with Agent Engine APIs
- **Apigee**: API gateway for external tool access (NCBI, UniProt, PDB). Rate limiting, caching, transformation

### Observability/evaluation
- **Cloud Trace**: Distributed tracing via OpenTelemetry. Visualize agent decision flow, tool calls, latency bottlenecks
- **Cloud Logging**: Structured logs from all agents. Log Explorer for debugging, log-based metrics
- **Cloud Monitoring**: Dashboards, alerts. Track token usage, error rates, SLO compliance
- **Vertex AI Evaluation**: Trajectory evaluation (expected vs actual steps), tool accuracy, intent resolution. LLM-as-judge scorers

**Alternative observability**: LangSmith (if using LangGraph), Arize Phoenix (component-based eval), W&B Weave (agent-specific). All integrate via OpenTelemetry.

## Workflow implementations

### Literature search + synthesis
```python
# ADK Literature Agent
from vertexai import agent_engines
from google.adk import agents, tools

@tools.tool
def pubmed_search(query: str, max_results: int = 50) -> list[dict]:
    """Search PubMed via NCBI E-utilities API"""
    # Implementation via MCP server on Cloud Run
    return mcp_client.call_tool("pubmed_search", {"query": query})

@tools.tool
def semantic_search(query: str, filters: dict) -> list[dict]:
    """Semantic search over Vertex Vector DB"""
    embeddings = vertex_ai.TextEmbeddingModel("textembedding-gecko@003")
    query_embedding = embeddings.get_embeddings([query])[0].values
    results = vector_search.query(
        query_embedding=query_embedding,
        num_neighbors=20,
        filters=filters  # publication_date, journal, study_type
    )
    return results

literature_agent = agents.LanggraphAgent(
    model="gemini-2.5-pro",
    tools=[pubmed_search, semantic_search, extract_citations, synthesize_findings],
    system_prompt="""You are a biomedical literature expert. 
    Search comprehensively, prioritize recent high-impact studies, 
    synthesize findings with proper citations."""
)

# Deploy to Agent Engine
remote_agent = agent_engines.create(
    literature_agent,
    display_name="literature-agent",
    requirements=["google-cloud-aiplatform[agent_engines]", "biopython"],
    extra_packages=[]
)
```

**Architecture**: Agent Engine runtime → calls MCP server (Cloud Run) → E-utils API. RAG via Vertex Vector Search (10M paper embeddings, MedCPT). Synthesis via Gemini with citation extraction. Output stored in Firestore for downstream agents.

### Experiment design
```python
# Reflection loop pattern (Google co-scientist)
from google.adk.agents import Agent, WorkflowAgent

@tools.tool
def generate_protocol(hypothesis: dict, constraints: dict) -> dict:
    """Generate experimental protocol"""
    prompt = f"Design experiment for: {hypothesis['description']}\n"
    prompt += f"Constraints: budget=${constraints['budget']}, "
    prompt += f"timeline={constraints['weeks']}w, equipment={constraints['equipment']}"
    
    response = gemini_model.generate_content(prompt)
    return parse_protocol(response.text)

@tools.tool
def critique_protocol(protocol: dict) -> dict:
    """Reflection agent reviews protocol"""
    critique_prompt = f"Review this protocol for feasibility, controls, statistics:\n{protocol}"
    return gemini_model.generate_content(critique_prompt)

@tools.tool
def refine_protocol(protocol: dict, critique: dict) -> dict:
    """Evolution agent improves protocol"""
    # Iterative refinement logic
    pass

design_workflow = WorkflowAgent(
    agents=[
        ("generate", generate_protocol),
        ("critique", critique_protocol),
        ("refine", refine_protocol)
    ],
    workflow_type="loop",  # Iterate until quality threshold
    max_iterations=3
)
```

**Pattern**: Generation → Reflection → Ranking → Evolution (Google's proven approach). Supervisor decides when protocol quality sufficient. Store intermediate versions for provenance.

### Bioinformatics pipelines
```python
# MCP server for bioinformatics tools (Cloud Run)
from mcp import Server, Tool
import subprocess

server = Server("bioinformatics-tools")

@server.tool()
def run_blast(sequence: str, database: str, e_value: float = 0.001) -> dict:
    """BLAST sequence search"""
    # Containerized BLAST execution
    result = subprocess.run([
        "blastp", "-query", sequence, 
        "-db", database, "-evalue", str(e_value),
        "-outfmt", "6"  # Tabular output
    ], capture_output=True)
    return parse_blast_output(result.stdout)

@server.tool()
def run_nextflow_pipeline(workflow: str, params: dict) -> str:
    """Execute nf-core pipeline"""
    # Trigger Cloud Batch job or GKE pod
    job_id = batch_client.create_job(
        container="nfcore/rnaseq:3.14.0",
        command=["nextflow", "run", workflow, "--params-file", params],
        machine_type="n2-standard-8",
        disk_size_gb=500
    )
    return job_id

# Deploy MCP server
gcloud run deploy bioinformatics-mcp \
  --source . \
  --region us-central1 \
  --cpu 4 --memory 16Gi \
  --timeout 3600 \
  --max-instances 10
```

**Agent calls MCP server**:
```python
bioinformatics_agent = agents.Agent(
    model="gemini-2.5-flash",  # Fast, cheap for tool orchestration
    tools=mcp.get_tools("https://bioinformatics-mcp-xxx.run.app"),
    system_prompt="You orchestrate bioinformatics workflows. Parse user intent, select tools, handle errors."
)

# Supervisor routes genomics queries here
response = bioinformatics_agent.query(
    "Run differential expression analysis on GEO dataset GSE12345, compare tumor vs normal"
)
```

**Advanced**: BioAgents pattern - fine-tune Phi-3 on Biocontainers tool docs + nf-core workflows. RAG on EDAM ontology. 10x cheaper than Gemini for routine tasks, similar accuracy.

### Clinical review
```python
# FHIR-compliant clinical agent
@tools.tool
def query_ehr(patient_id: str, data_types: list[str]) -> dict:
    """Retrieve patient data via FHIR API"""
    fhir_client = FHIRClient(settings={"app_id": "mayo-agents"})
    bundle = fhir_client.get(
        f"Patient/{patient_id}/$everything",
        params={"_type": ",".join(data_types)}
    )
    return bundle.as_json()

@tools.tool
def check_drug_interactions(medications: list[str]) -> dict:
    """Query drug interaction database"""
    # Call RxNorm/FDA API via MCP
    pass

clinical_agent = agents.LanggraphAgent(
    model="gemini-2.5-pro",  # Need reasoning for complex cases
    tools=[query_ehr, check_drug_interactions, assess_risk, generate_report],
    system_prompt="""You are a clinical decision support agent. 
    Analyze patient data, identify risks, provide evidence-based recommendations.
    ALWAYS cite guidelines and studies. Flag high-risk findings for human review.""",
    guardrails={
        "require_human_review_for": ["high_risk_medication", "off_label_use"],
        "max_confidence_threshold": 0.85  # Require HITL below this
    }
)
```

**HIPAA compliance**: 
- Agent Engine with CMEK (customer-managed encryption keys)
- VPC Service Controls (private traffic only)
- Audit logging (all agent actions logged to BigQuery)
- PHI redaction in traces (Cloud DLP integration)
- Session isolation per patient (Memory Bank)

### Data analysis + PyTorch integration
```python
# Analysis agent triggers training jobs
@tools.tool
def train_genomics_model(dataset_uri: str, config: dict) -> str:
    """Trigger PyTorch training on Vertex AI"""
    from google.cloud import aiplatform
    
    job = aiplatform.CustomTrainingJob(
        display_name="genomics-model-training",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest",
        requirements=["pytorch-lightning", "scanpy", "torch_geometric"]
    )
    
    model = job.run(
        dataset=dataset_uri,
        args=[
            "--model_type", config["architecture"],
            "--learning_rate", str(config["lr"]),
            "--batch_size", str(config["batch_size"]),
            "--epochs", str(config["epochs"])
        ],
        replica_count=1,
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_V100",
        accelerator_count=2  # Multi-GPU DDP
    )
    
    return model.resource_name

@tools.tool  
def monitor_training(job_id: str) -> dict:
    """Check training job status + metrics"""
    job = aiplatform.CustomTrainingJob(job_name=job_id)
    if job.state == "SUCCEEDED":
        return {"status": "complete", "metrics": get_tensorboard_metrics(job_id)}
    return {"status": job.state.name, "progress": estimate_progress(job)}

# Agent workflow
analysis_agent.query("Train a graph neural network on this single-cell RNA-seq dataset for cell type prediction")
# Agent autonomously: parses intent → prepares dataset → selects architecture → trains → evaluates → deploys endpoint
```

**Serving pattern**:
```python
# Deploy trained model as Vertex AI Endpoint
endpoint = aiplatform.Endpoint.create(display_name="genomics-classifier")
model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    min_replica_count=1,
    max_replica_count=10,
    traffic_percentage=100
)

# Wrap as agent tool
@tools.tool
def predict_cell_type(gene_expression: list[float]) -> dict:
    """Classify cell type from expression profile"""
    prediction = endpoint.predict(instances=[gene_expression])
    return {"cell_type": prediction[0]["class"], "confidence": prediction[0]["score"]}
```

**MLOps integration**: MLflow autologging + Vertex Experiments for tracking. W&B Weave for agent-level observability (traces include model calls).

### Hypothesis generation
```python
# Knowledge graph + LLM reasoning
@tools.tool
def query_knowledge_graph(entities: list[str], relation_types: list[str]) -> list[dict]:
    """Query PrimeKG or custom biomedical KG"""
    # Neo4j/TigerGraph via MCP server
    cypher_query = f"""
    MATCH (e1)-[r:{"|".join(relation_types)}]->(e2)
    WHERE e1.name IN {entities}
    RETURN e1, r, e2, r.evidence
    LIMIT 100
    """
    return graph_db.run(cypher_query)

@tools.tool
def literature_based_discovery(concept_a: str, concept_b: str) -> list[dict]:
    """Find bridging concepts (ABC algorithm)"""
    # Vector search for intermediate concepts
    pass

hypothesis_agent = agents.Agent(
    model="gemini-2.5-pro-experimental",  # Enhanced reasoning
    tools=[query_knowledge_graph, literature_based_discovery, assess_novelty, propose_mechanism],
    system_prompt="""Generate novel, testable hypotheses by:
    1. Analyzing knowledge graph patterns
    2. Identifying gaps in literature  
    3. Proposing mechanistic explanations
    4. Assessing feasibility and impact"""
)

# Multi-agent debate for hypothesis refinement (Google co-scientist pattern)
debate_system = WorkflowAgent([
    ("generate", hypothesis_agent),
    ("critique", reflection_agent),
    ("rank", ranking_agent),
    ("evolve", evolution_agent)
], workflow_type="loop")
```

**Output**: Ranked hypotheses with mechanistic explanations, literature support, feasibility scores. Top 3 sent to human researcher for experimental validation.

## Multi-agent communication patterns

### Supervisor orchestration (recommended)
```python
from google.adk.agents import Agent, WorkflowAgent
from google.adk.types import Message

# Supervisor with dynamic routing
class ResearchSupervisor(Agent):
    def __init__(self):
        self.workers = {
            "literature": literature_agent,
            "experiment": design_agent,
            "bioinformatics": bioinformatics_agent,
            "clinical": clinical_agent,
            "analysis": data_analysis_agent,
            "hypothesis": hypothesis_agent
        }
    
    async def route_task(self, state: dict) -> str:
        """LLM-based routing with forced function calling"""
        routing_prompt = f"""
        User query: {state['query']}
        Available agents: {list(self.workers.keys())}
        
        Which agent should handle this? Consider:
        - Query type and domain
        - Required expertise
        - Agent dependencies
        """
        
        response = await self.model.generate_content(
            routing_prompt,
            tools=[{"name": "route_to_agent", "parameters": {"agent": "string"}}],
            tool_config={"function_calling_config": {"mode": "ANY"}}  # Force tool use
        )
        
        return response.tool_calls[0].args["agent"]
    
    async def orchestrate(self, query: str) -> dict:
        state = {"query": query, "results": {}, "history": []}
        
        while not self.is_complete(state):
            next_agent = await self.route_task(state)
            result = await self.workers[next_agent].query(state)
            state["results"][next_agent] = result
            state["history"].append({"agent": next_agent, "result": result})
            
            # Check if human review needed
            if result.get("requires_human_review"):
                state = await self.human_in_loop_checkpoint(state)
        
        return self.synthesize_results(state)
```

**Deployment**:
```bash
# Single command deployment via ADK CLI
adk deploy agent-engine \
  --project mayo-clinic-agents \
  --location us-central1 \
  --agent-path ./supervisor_agent \
  --requirements requirements.txt
```

### Agent-to-Agent (A2A) protocol
```python
# Distributed agents communicate via A2A
# Agent 1 (deployed on Agent Engine)
from google.adk.a2a import AgentClient

bioinformatics_client = AgentClient(
    url="https://bioinformatics-agent-xxx.vertex-ai.goog",
    auth=google.auth.default()
)

# Call remote agent
task_id = bioinformatics_client.invoke_async(
    method="run_pipeline",
    args={"workflow": "nf-core/rnaseq", "dataset": "gs://bucket/data"}
)

# Poll for results (with timeout)
result = bioinformatics_client.wait_for_result(task_id, timeout=3600)

# Agent 2 (separate deployment, possibly different framework)
# Registers as A2A-compatible agent
from google.adk.a2a import AgentServer

server = AgentServer(
    agent=bioinformatics_agent,
    port=8080,
    agent_card={
        "name": "bioinformatics-pipeline-agent",
        "capabilities": ["rna_seq", "chip_seq", "single_cell"],
        "protocols": ["A2A/1.0"]
    }
)
```

**Use case**: Mayo Clinic + external institution collaboration. Different orgs run specialized agents, communicate via A2A (secure, authenticated, standardized).

### Parallel execution with aggregation
```python
# Fan-out/fan-in for multi-omics analysis
async def multi_omics_analysis(sample_id: str) -> dict:
    # Parallel agent calls
    tasks = [
        genomics_agent.analyze(sample_id, "WGS"),
        transcriptomics_agent.analyze(sample_id, "RNA-seq"),
        proteomics_agent.analyze(sample_id, "mass-spec"),
        metabolomics_agent.analyze(sample_id, "metabolomics")
    ]
    
    results = await asyncio.gather(*tasks)  # Wait for all
    
    # Integration agent synthesizes
    integrated = integration_agent.synthesize({
        "genomics": results[0],
        "transcriptomics": results[1],
        "proteomics": results[2],
        "metabolomics": results[3]
    })
    
    return integrated
```

**Implementation**: Cloud Workflows for orchestration, Pub/Sub for async coordination, Firestore for shared state.

## Memory architecture

**3-tier memory system**:

**Short-term (session)**: Firestore documents. Agent Engine Memory Bank (managed service, native integration). 5-20 most recent messages per session. TTL: 24 hours.

**Long-term (persistent)**: Vertex AI Vector Search for semantic memory. Stores: user preferences, past research findings, domain knowledge. Retrieval: hybrid search (vector similarity + metadata filters). **Embeddings**: textembedding-gecko@003 (768-dim, optimized for scientific text).

**Working memory (scratchpad)**: Redis (Memorystore). Agent notes, intermediate computations, tool call results. Expiration: 1 hour. Pattern: `agent_id:session_id:note_type` keys.

```python
# Memory integration in agents
from google.adk.memory import MemoryBank

agent = agents.LanggraphAgent(
    model="gemini-2.5-pro",
    tools=[...],
    memory=MemoryBank(
        short_term="firestore",  # Managed by Agent Engine
        long_term=VectorSearchMemory(
            index_name="research-knowledge",
            embedding_model="textembedding-gecko@003"
        ),
        working=RedisMemory(
            host="10.0.0.3",  # Memorystore instance
            ttl=3600
        )
    )
)

# Automatic memory operations
response = agent.query("What were the results from our last AML study?")
# Agent: 
# 1. Checks short-term (recent conversation)
# 2. Queries long-term (semantic search for "AML study results")  
# 3. Uses working memory for reasoning scratchpad
# 4. Synthesizes answer + stores new findings in long-term
```

## Observability stack

**Cloud Trace + OpenTelemetry**: Distributed tracing across all agents. Visualize full request flow: user query → supervisor → workers → tools → LLMs.

```python
# Automatic tracing with ADK
agent = agents.Agent(
    model="gemini-2.5-pro",
    tools=[...],
    observability={
        "trace_to_cloud": True,  # Export to Cloud Trace
        "trace_provider": "opentelemetry",
        "sampling_rate": 1.0  # 100% during dev, 0.1 in prod
    }
)

# Custom spans for important operations
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

@tools.tool
def complex_analysis(data: dict) -> dict:
    with tracer.start_as_current_span("complex_analysis") as span:
        span.set_attribute("data_size", len(data))
        span.set_attribute("analysis_type", "differential_expression")
        
        result = perform_analysis(data)
        
        span.set_attribute("result_size", len(result))
        span.add_event("Analysis complete", {"genes_found": result["count"]})
        
        return result
```

**Dashboards**: Cloud Monitoring with custom dashboards. Key metrics:
- Agent invocation rate (requests/min)
- Latency percentiles (p50, p95, p99) per agent
- Token usage + cost (per agent, per model)
- Error rate (by agent, by tool)
- Tool call success rate
- Human-in-loop intervention rate

**Alerting**: 
- Error rate > 5% for 5 min → page on-call
- P95 latency > 30s → investigate
- Daily token budget exceeded → notify + throttle
- Agent returning hallucinations (detected by evaluation) → circuit breaker

**Alternative**: Arize Phoenix for deep agent debugging. Deploy Phoenix UI on Cloud Run, integrate via OpenTelemetry. Component-based evaluation (router accuracy, tool selection, retrieval quality).

## Evaluation framework

**Continuous evaluation pipeline**:

```python
# Evaluation dataset management
from google.cloud import bigquery
from vertexai.preview.evaluation import EvaluationPipeline

eval_dataset = [
    {
        "query": "Design experiment to test EGFR inhibitor in lung cancer cell lines",
        "expected_tools": ["literature_search", "protocol_generator", "resource_estimator"],
        "expected_steps": 5,
        "quality_criteria": ["has_controls", "statistical_power", "cited_precedent"]
    },
    # ... 100+ test cases covering all workflows
]

# Store in BigQuery for version control + sharing
bq_client.load_table_from_json(eval_dataset, "mayo_agents.evaluation_cases")

# Evaluation pipeline (runs on every deployment)
pipeline = EvaluationPipeline(
    agent=supervisor_agent,
    dataset=eval_dataset,
    metrics=[
        "trajectory_accuracy",  # Did agent follow expected path?
        "tool_call_correctness",  # Right tools, right params?
        "output_quality",  # LLM-as-judge with Gemini 2.0
        "latency",
        "cost"
    ],
    judges={
        "output_quality": gemini_judge_config(
            model="gemini-2.0-flash-exp",
            criteria="Evaluate experiment design for: scientific rigor, feasibility, novelty"
        )
    }
)

results = pipeline.run()

# Quality gate: block deployment if metrics regress
if results["trajectory_accuracy"] < 0.85 or results["output_quality"] < 0.80:
    raise DeploymentBlockedError("Agent quality below threshold")
```

**Production monitoring**: Sample 10% of traffic, run evaluation asynchronously, track metrics over time in BigQuery. Alert on degradation.

**Human feedback loop**: Allow researchers to rate agent responses (👍👎). Store ratings → add to eval dataset → retrain/refine prompts.

## Human-in-the-loop implementation

```python
# LangGraph checkpointing for HITL
from google.adk.hitl import HumanApprovalRequired
from google.adk.agents import LanggraphAgent

agent = LanggraphAgent(
    model="gemini-2.5-pro",
    tools=[run_expensive_assay, order_reagents, submit_irb_protocol],
    checkpointer="firestore",  # Persistent state
    interrupt_before=["run_expensive_assay", "submit_irb_protocol"],  # Require approval
)

# Agent execution pauses at checkpoint
try:
    result = agent.invoke({"query": "Run full genomic screen on 500 patient samples"})
except HumanApprovalRequired as e:
    # Send notification (Slack, email, UI)
    send_approval_request(
        user="researcher@mayo.edu",
        checkpoint_id=e.checkpoint_id,
        planned_action=e.action,
        estimated_cost=e.metadata["cost"],
        review_url=f"https://agents.mayo.edu/review/{e.checkpoint_id}"
    )
    
    # User reviews in web UI
    # On approval, resume:
    result = agent.resume(checkpoint_id=e.checkpoint_id, approved=True)
```

**HITL triggers**:
- High-cost operations (>$1000)
- IRB/regulatory submissions
- Clinical recommendations (always require MD review)
- Low confidence predictions (<0.7)
- Novel hypotheses (no literature precedent)

## Implementation roadmap

### Phase 1: Foundation (Weeks 1-4)
**Goal**: Single-agent MVP with core infrastructure

**Tasks**:
- Set up GCP project with Vertex AI, Agent Engine, Cloud Run
- Deploy Literature Agent (ADK + Vertex Vector Search for PubMed)
- Build MCP server for PubMed/ArXiv APIs (Cloud Run)
- Firestore for state, Cloud SQL for metadata
- Cloud Trace integration for observability
- Basic evaluation dataset (20 literature queries)

**Deliverable**: Working literature search agent accessible via API. Researcher enters query → agent searches → synthesizes → returns summary with citations.

**Code**:
```bash
# Project setup
gcloud config set project mayo-clinic-agents
gcloud services enable aiplatform.googleapis.com run.googleapis.com

# Deploy literature agent
cd agents/literature
adk deploy agent-engine \
  --agent-path . \
  --requirements requirements.txt \
  --display-name literature-agent

# Deploy MCP server
cd tools/pubmed-mcp
gcloud run deploy pubmed-mcp --source . --region us-central1
```

**Validate**: Run 20 test queries, compare agent output to human-curated gold standard. Target: >0.8 citation accuracy, <30s latency.

### Phase 2: Multi-agent system (Weeks 5-8)
**Goal**: Deploy all 6 agents with supervisor orchestration

**Tasks**:
- Deploy remaining 5 agents (Experiment, Bioinformatics, Clinical, Analysis, Hypothesis)
- Build supervisor agent with LLM-based routing
- Implement A2A protocol for agent communication
- Cloud Workflows for complex orchestrations (multi-omics analysis)
- Pub/Sub for async agent coordination
- Memory Bank (Agent Engine) for session management
- Evaluation pipeline (100 test cases across all workflows)

**Deliverable**: End-to-end workflow. Researcher: "Design and execute RNA-seq experiment comparing treatment vs control" → supervisor routes through Literature → Hypothesis → Experiment → Bioinformatics → Analysis → delivers results.

**Code**:
```python
# Supervisor deployment
supervisor = WorkflowAgent(
    agents={
        "literature": literature_agent_url,
        "experiment": experiment_agent_url,
        "bioinformatics": bioinformatics_agent_url,
        "clinical": clinical_agent_url,
        "analysis": analysis_agent_url,
        "hypothesis": hypothesis_agent_url
    },
    routing_strategy="llm_based",  # Gemini decides routing
    state_backend="firestore"
)

adk deploy agent-engine --agent-path ./supervisor
```

**Validate**: 50 end-to-end workflows. Track: task completion rate (>90%), routing accuracy (>85%), total latency (<5 min for literature+design, <2 hours for bioinformatics).

### Phase 3: PyTorch integration (Weeks 9-10)
**Goal**: Agents trigger and manage ML training

**Tasks**:
- Vertex AI Training job templates (PyTorch, PyTorch Lightning)
- Data Analysis Agent tools: `train_model()`, `monitor_job()`, `deploy_endpoint()`
- MLflow integration for experiment tracking
- Model registry (Vertex AI Model Registry)
- Auto-deployment pipeline (train → evaluate → deploy if metrics improve)

**Deliverable**: Agent autonomously trains genomics classifier, evaluates, deploys to endpoint, uses for inference in downstream tasks.

**Code**:
```python
# Analysis agent with training capability
@tools.tool
def train_and_deploy(dataset: str, config: dict) -> str:
    # Trigger training (as shown earlier)
    job = vertex_ai.CustomTrainingJob(...).run(...)
    
    # Evaluate
    metrics = evaluate_model(job.model_resource_name, validation_dataset)
    
    # Deploy if better than current production
    if metrics["accuracy"] > current_production_metrics["accuracy"]:
        endpoint = deploy_to_vertex(job.model_resource_name)
        return f"Deployed to {endpoint.resource_name}"
    else:
        return "Model did not improve, keeping current production version"
```

**Validate**: Train 5 genomics models (cell type classification, variant effect prediction, gene expression modeling). Verify agents handle failures, track experiments, deploy correctly.

### Phase 4: Production hardening (Weeks 11-12)
**Goal**: HIPAA compliance, security, reliability, cost controls

**Tasks**:
- CMEK for all data (Firestore, Cloud SQL, GCS, Vector Search)
- VPC Service Controls (private network, no public internet)
- Audit logging to BigQuery (all agent actions, LLM prompts/responses)
- Cloud DLP for PHI redaction in logs/traces
- Circuit breakers for all agents (pybreaker)
- Rate limiting (10 requests/min per user)
- Cost monitoring dashboards (daily budget alerts)
- HITL checkpoints for high-risk operations
- Comprehensive alerting (PagerDuty integration)

**Deliverable**: Production-ready system with security attestation, <0.1% error rate, cost controls.

**Code**:
```python
# Circuit breaker pattern
from pybreaker import CircuitBreaker

clinical_breaker = CircuitBreaker(fail_max=5, timeout_duration=60)

@clinical_breaker
@tools.tool
def query_ehr(patient_id: str) -> dict:
    # If 5 consecutive failures, breaker opens for 60s
    return fhir_client.get(f"Patient/{patient_id}")

# Cost controls
from google.cloud import billing_budgets_v1

budget = billing_budgets_v1.Budget(
    display_name="agents-daily-budget",
    budget_filter=billing_budgets_v1.Filter(
        projects=["projects/mayo-clinic-agents"]
    ),
    amount=billing_budgets_v1.BudgetAmount(
        specified_amount={"units": 500}  # $500/day
    ),
    threshold_rules=[
        billing_budgets_v1.ThresholdRule(threshold_percent=0.5),  # Alert at 50%
        billing_budgets_v1.ThresholdRule(threshold_percent=0.9),  # Alert at 90%
        billing_budgets_v1.ThresholdRule(threshold_percent=1.0),  # Page at 100%
    ]
)
```

**Validate**: Penetration testing, HIPAA audit, load testing (100 concurrent users), chaos engineering (kill random agents, verify recovery).

### Phase 5: Continuous improvement (Ongoing)
**Goal**: Iterate based on researcher feedback

**Tasks**:
- Collect user feedback (ratings, qualitative comments)
- Expand evaluation dataset (add new test cases monthly)
- Monitor quality metrics (trajectory accuracy, output quality)
- A/B test prompt variations, model versions
- Fine-tune models on Mayo-specific data (with appropriate governance)
- Add new tools/agents as needed
- Quarterly security reviews

## Code patterns library

### Agent tool template
```python
from google.adk import tools
from opentelemetry import trace
import mlflow

tracer = trace.get_tracer(__name__)

@tools.tool
@tracer.start_as_current_span("tool_name")
def my_analysis_tool(param1: str, param2: int) -> dict:
    """
    Clear description for LLM. Explain what it does, when to use it.
    
    Args:
        param1: Description with type info
        param2: Another parameter
        
    Returns:
        dict with keys: result, confidence, metadata
    """
    # Log to MLflow for tracking
    with mlflow.start_run(nested=True):
        mlflow.log_param("param1", param1)
        mlflow.log_param("param2", param2)
        
        # Tool implementation
        result = perform_computation(param1, param2)
        
        # Add trace metadata
        span = trace.get_current_span()
        span.set_attribute("result_size", len(result))
        span.add_event("computation_complete")
        
        mlflow.log_metric("result_count", len(result))
        
        return {
            "result": result,
            "confidence": 0.95,
            "metadata": {"source": "mayo_db", "timestamp": datetime.now()}
        }
```

### Error handling pattern
```python
from tenacity import retry, stop_after_attempt, wait_exponential
from pybreaker import CircuitBreaker

breaker = CircuitBreaker(fail_max=5, timeout_duration=60)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
@breaker
@tools.tool
def external_api_call(query: str) -> dict:
    """Resilient external API call with retries + circuit breaker"""
    try:
        response = requests.post(
            API_ENDPOINT,
            json={"query": query},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
        
    except requests.Timeout:
        logger.error(f"Timeout calling API: {query}")
        # Fallback to cached data if available
        cached = redis_client.get(f"cache:{query}")
        if cached:
            return json.loads(cached)
        raise
        
    except requests.HTTPError as e:
        if e.response.status_code == 429:  # Rate limited
            logger.warning("Rate limited, backing off")
            raise  # tenacity will retry with exponential backoff
        else:
            logger.error(f"API error {e.response.status_code}: {e}")
            raise
```

### PyTorch training orchestration
```python
@tools.tool
async def orchestrate_training_pipeline(
    dataset_uri: str,
    model_config: dict,
    evaluation_metrics: list[str]
) -> dict:
    """Full ML training pipeline with agent orchestration"""
    
    # 1. Prepare data (via Cloud Function)
    prep_job = trigger_data_prep(dataset_uri)
    await wait_for_completion(prep_job, timeout=600)
    
    # 2. Trigger training on Vertex AI
    training_job = aiplatform.CustomTrainingJob(
        display_name=f"agent-training-{timestamp}",
        container_uri="us-docker.pkg.dev/mayo-agents/training/pytorch:latest",
        model_serving_container_image_uri="us-docker.pkg.dev/mayo-agents/serving/pytorch:latest"
    ).run(
        dataset=prep_job.output_dataset,
        args=[f"--{k}={v}" for k, v in model_config.items()],
        replica_count=1,
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_V100",
        accelerator_count=2,
        sync=False  # Async execution
    )
    
    # 3. Monitor with polling
    while training_job.state not in ["SUCCEEDED", "FAILED", "CANCELLED"]:
        await asyncio.sleep(60)
        
        # Log progress to agent state
        metrics = get_tensorboard_metrics(training_job.resource_name)
        logger.info(f"Training progress: {metrics}")
        
        # Check for divergence
        if metrics.get("loss") == float("inf"):
            training_job.cancel()
            return {"status": "failed", "reason": "training_diverged"}
    
    if training_job.state != "SUCCEEDED":
        return {"status": "failed", "reason": training_job.error.message}
    
    # 4. Evaluate on held-out set
    eval_results = evaluate_model(
        model_resource_name=training_job.model_resource_name,
        test_dataset=prep_job.test_dataset,
        metrics=evaluation_metrics
    )
    
    # 5. Deploy if meets quality threshold
    if all(eval_results[m] > THRESHOLDS[m] for m in evaluation_metrics):
        endpoint = deploy_model_to_endpoint(training_job.model_resource_name)
        
        return {
            "status": "success",
            "model_resource_name": training_job.model_resource_name,
            "endpoint_url": endpoint.resource_name,
            "metrics": eval_results
        }
    else:
        return {
            "status": "model_below_threshold",
            "metrics": eval_results,
            "thresholds": THRESHOLDS
        }
```

### MCP server template (Cloud Run)
```python
# bioinformatics_tools_mcp.py
from mcp.server import MCPServer
from google.cloud import storage
import subprocess

app = MCPServer(name="bioinformatics-tools")

@app.tool()
async def run_blast_search(
    sequence: str,
    database: str = "nr",
    e_value: float = 0.001
) -> dict:
    """BLAST protein sequence search"""
    
    # Write sequence to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as f:
        f.write(f">query\n{sequence}\n")
        f.flush()
        
        # Run BLAST (containerized binary)
        result = subprocess.run([
            "blastp",
            "-query", f.name,
            "-db", f"/data/blast/{database}",
            "-evalue", str(e_value),
            "-outfmt", "6 qseqid sseqid pident length evalue bitscore stitle",
            "-max_target_seqs", "10"
        ], capture_output=True, text=True, check=True)
    
    # Parse results
    hits = []
    for line in result.stdout.strip().split('\n'):
        if line:
            fields = line.split('\t')
            hits.append({
                "subject_id": fields[1],
                "identity_pct": float(fields[2]),
                "length": int(fields[3]),
                "e_value": float(fields[4]),
                "bit_score": float(fields[5]),
                "description": fields[6] if len(fields) > 6 else ""
            })
    
    return {"hits": hits, "num_hits": len(hits)}

@app.tool()
async def fetch_protein_structure(uniprot_id: str) -> dict:
    """Fetch protein structure from AlphaFold DB"""
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    response = requests.get(url)
    
    if response.status_code == 200:
        # Upload to GCS for persistence
        bucket = storage.Client().bucket("mayo-structures")
        blob = bucket.blob(f"{uniprot_id}.pdb")
        blob.upload_from_string(response.text)
        
        return {
            "uniprot_id": uniprot_id,
            "pdb_url": blob.public_url,
            "confidence": extract_plddt_scores(response.text)
        }
    else:
        return {"error": "Structure not found", "uniprot_id": uniprot_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

**Deployment**:
```dockerfile
# Dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y blast2 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY bioinformatics_tools_mcp.py .
COPY blast_dbs /data/blast/
CMD ["python", "bioinformatics_tools_mcp.py"]
```

```bash
gcloud run deploy bioinformatics-mcp \
  --source . \
  --region us-central1 \
  --cpu 4 --memory 16Gi --timeout 3600 \
  --max-instances 20 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=mayo-clinic-agents
```

## Framework and library recommendations

**Agent framework**: **Agent Development Kit (ADK)** - Google's purpose-built framework. Native Vertex AI integration, <100 lines per agent, A2A/MCP support, single-command deployment. Alternatives: LangGraph (more control, harder deployment), CrewAI (simpler, less flexible).

**Orchestration**: **Vertex AI Agent Engine** for agent runtime + **Cloud Workflows** for complex multi-step processes. Alternative: Temporal (durable execution, better for long-running workflows spanning days).

**LLMs**: **Gemini 2.5 Pro** (complex reasoning, experiment design, hypothesis generation), **Gemini 2.5 Flash** (fast tool routing, bioinformatics orchestration). Consider **fine-tuned Phi-3** for specialized tasks (cost reduction).

**Embeddings**: **textembedding-gecko@003** (768-dim, optimized for scientific text). Alternative: domain-specific like **MedCPT** or **PubMedBERT** embeddings (better for biomedical literature, requires custom deployment).

**Vector DB**: **Vertex AI Vector Search** (managed, auto-scaling, sub-50ms latency, 10M+ vectors). Alternative: **Pinecone** (more mature tooling), **Weaviate** (hybrid search, better filtering).

**Observability**: **Cloud Trace + Cloud Monitoring** (native GCP integration, no extra cost). Alternative: **LangSmith** (better agent-specific debugging, paid), **Arize Phoenix** (component-based eval, open-source).

**Evaluation**: **Vertex AI Evaluation** (trajectory eval, LLM-as-judge, integrated with Agent Engine). Alternative: **Arize Phoenix** (detailed agent debugging), **DeepEval** (custom metrics).

**MLOps**: **MLflow** (experiment tracking, model registry, deployment). Integrates with Vertex AI. Alternative: **W&B** (better visualization, agent tracing via Weave).

**HITL**: **LangGraph checkpointing** (native to ADK). Alternative: **Temporal** (production-grade workflow orchestration), **HumanLayer SDK** (async approval via Slack/email).

## Cost projections

**Gemini API** (primary cost driver):
- Gemini 2.5 Pro: $1.25/M input tokens, $5/M output tokens
- Gemini 2.5 Flash: $0.075/M input, $0.30/M output
- Typical workflow: 10K input + 2K output per agent invocation
- Cost per workflow (6 agents): ~$0.10 with Flash, ~$0.20 with Pro

**Agent Engine**: $0.02/vCPU-hour + $0.0025/GiB-hour. Typical: 1 vCPU, 4 GiB per agent. 6 agents running 24/7 = $1,440/month. Actual usage likely 10-20% = ~$300/month.

**Vector Search**: $0.096/GB/month storage + $0.0001/1K queries. 10M embeddings (768-dim) = ~30GB = $3/month + query costs (~$50/month for 500K queries).

**Cloud Run** (MCP servers): Pay per request. Typical: $0.01/100K requests. 100K tool calls/month = $1.

**Vertex AI Training**: V100 GPU = $2.48/hour. T4 = $0.35/hour. 10 training jobs/week @ 4 hours each = ~$100/week GPU costs.

**Total estimated monthly cost**:
- Development (low traffic): $500-1000
- Production (1000 queries/day): $3000-5000
- Heavy usage (10K queries/day): $20K-30K

**Cost optimization**:
- Use Flash for 80% of tasks, Pro for complex reasoning only
- Cache tool results (Redis) for 24 hours
- Fine-tune small models (Phi-3) for routine bioinformatics tasks
- Batch training jobs, use preemptible instances
- Set daily budgets + alerts

## Security considerations

**Data protection**:
- CMEK for all storage (Firestore, Cloud SQL, GCS, Vector Search)
- VPC-SC perimeter around all resources
- Private Service Connect for Vertex AI Agent Engine
- PHI redaction via Cloud DLP in logs/traces

**Access control**:
- Workload Identity for service accounts (no keys)
- Agent-specific service accounts (least privilege)
- IAM conditions for fine-grained access
- Audit logging to BigQuery (immutable)

**Compliance**:
- HIPAA-compliant deployment (Agent Engine + supporting services certified)
- BAA with Google Cloud
- Encrypted in transit (TLS 1.3) and at rest (AES-256)
- Data residency controls (us-central1 only)

**Threat detection**:
- Security Command Center: Agent Engine Threat Detection (preview)
- Cloud Armor for DDoS protection
- Anomaly detection on agent behavior (sudden spike in API calls, unusual query patterns)

## Key takeaways

**Architecture**: Supervisor pattern with 6 specialized agents. Vertex AI Agent Engine (runtime) + ADK (framework) + Cloud Run (tools).

**GCP services**: Agent Engine, Cloud Run, Vertex AI (training, serving, vector search), Firestore, Cloud SQL, Pub/Sub, Cloud Trace/Monitoring.

**Workflows**: All 7 workflows implementable with patterns from Google co-scientist + NVIDIA BioNeMo + research papers (BioAgents, GeneAgent, etc).

**PyTorch**: Agents trigger Vertex AI Training jobs, monitor via API, auto-deploy models to Vertex AI Endpoints. MLflow for tracking.

**Memory**: 3-tier (short-term Firestore, long-term Vector Search, working Redis). Agent Engine Memory Bank for managed sessions.

**Observability**: Cloud Trace (OpenTelemetry), Cloud Monitoring, Cloud Logging. Alternative: LangSmith or Arize Phoenix for deep debugging.

**Evaluation**: Continuous eval pipeline. LLM-as-judge for output quality, trajectory comparison for agent behavior. Store cases in BigQuery.

**HITL**: LangGraph checkpointing for approval workflows. Firestore for persistent state across interruptions.

**Timeline**: 12 weeks to production. Phase 1 (foundation), Phase 2 (multi-agent), Phase 3 (PyTorch), Phase 4 (hardening), Phase 5 (iteration).

**Cost**: $500-1K/month dev, $3-5K/month production, $20-30K/month heavy usage. Primary driver: Gemini API calls.

**Validated approach**: Google's co-scientist validated in drug repurposing, target discovery, AMR mechanisms. NVIDIA BioNeMo used by Amgen, Genentech, Insilico. Patterns proven in production.

Start with Phase 1 (literature agent MVP in 4 weeks). Validate with researchers. Iterate rapidly. Add agents incrementally. Focus on human-AI collaboration, not full automation.