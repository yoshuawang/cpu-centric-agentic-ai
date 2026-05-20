"""
Batch web-LLM orchestrator using LangGraph batching with per-query NVTX markers.
Accepts multiple queries as CLI args, runs the full tool chain in a single batched graph invocation, and marks each node per query.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import os, math
import sys
import timeit
from typing import List, Dict, Optional
import nvtx
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from typing import TypedDict, List
from langgraph.graph import StateGraph
from langchain_core.runnables.config import RunnableConfig
import sys
import argparse
 
import time
from collections import defaultdict
from dataclasses import dataclass
import re
from urllib.parse import urlsplit, urlunsplit
 
# Global timing storage for statistics
timing_stats = defaultdict(list)

# Per-query trace: list of dicts, one per query, keyed by stage
query_traces: List[Dict[str, float]] = []
_trace_lock = __import__('threading').Lock()
 
# ----------------------------
# vLLM timing/metrics helpers
# ----------------------------

_PROM_HIST_SUM_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)_sum(?:\\{[^}]*\\})?\\s+(?P<value>[-+eE0-9\\.]+)\\s*$"
)
_PROM_HIST_COUNT_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)_count(?:\\{[^}]*\\})?\\s+(?P<value>[-+eE0-9\\.]+)\\s*$"
)


@dataclass(frozen=True)
class _PromHist:
    sum: float
    count: float


def _parse_prom_hist(metrics_text: str, metric_name: str) -> Optional[_PromHist]:
    """Parse a Prometheus histogram's _sum/_count (no labels) from text."""
    s = None
    c = None
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue
        m = _PROM_HIST_SUM_RE.match(line)
        if m and m.group("name") == metric_name:
            try:
                s = float(m.group("value"))
            except ValueError:
                s = None
            continue
        m = _PROM_HIST_COUNT_RE.match(line)
        if m and m.group("name") == metric_name:
            try:
                c = float(m.group("value"))
            except ValueError:
                c = None
            continue
    if s is None or c is None:
        return None
    return _PromHist(sum=s, count=c)


def _derive_vllm_metrics_url(openai_base_url: str) -> str:
    # Typical: http://host:port/v1  ->  http://host:port/metrics
    if openai_base_url.endswith("/"):
        openai_base_url = openai_base_url[:-1]
    if openai_base_url.endswith("/v1"):
        return openai_base_url[:-3] + "/metrics"
    return openai_base_url + "/metrics"


def _scrape_vllm_metrics(metrics_url: str, timeout_s: float = 3.0) -> Optional[str]:
    try:
        r = requests.get(metrics_url, timeout=timeout_s)
        r.raise_for_status()
        return r.text
    except Exception:
        return None


def _hist_delta_avg(before: Optional[_PromHist], after: Optional[_PromHist]) -> Optional[float]:
    if before is None or after is None:
        return None
    d_count = after.count - before.count
    d_sum = after.sum - before.sum
    if d_count <= 0:
        return None
    return d_sum / d_count


def _redact_url_userinfo(url: str) -> str:
    """Redact username/password from URLs before persisting run metadata."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.username and not parts.password:
        return url
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, f"<redacted>@{host}", parts.path, parts.query, parts.fragment))


def _run_metadata() -> Dict[str, object]:
    """Return non-secret run metadata suitable for JSON traces."""
    docker_keys = [
        "LANGCHAIN_RUN_ID",
        "LANGCHAIN_IMAGE",
        "LANGCHAIN_CONTAINER_NAME",
        "VLLM_IMAGE",
        "VLLM_CONTAINER_NAME",
        "DOCKER_NETWORK_NAME",
        "START_VLLM_CONTAINER",
        "VLLM_PORT",
    ]
    docker_metadata = {key.lower(): os.getenv(key) for key in docker_keys if os.getenv(key)}
    return {
        "vllm_openai_base_url": _redact_url_userinfo(os.getenv("VLLM_OPENAI_BASE_URL", "http://localhost:5000/v1")),
        "vllm_model": os.getenv("VLLM_MODEL", "openai/gpt-oss-20b"),
        "vllm_max_tokens": os.getenv("VLLM_MAX_TOKENS", "256"),
        "vllm_temperature": os.getenv("VLLM_TEMPERATURE", "0.0"),
        "tavily_api_key_present": bool(os.getenv("TAVILY_API_KEY")),
        "docker": docker_metadata,
    }


def _vllm_completion_stream(
    *,
    openai_base_url: str,
    model: str,
    prompt: str,
    timeout_s: float = 180.0,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[str, float, float]:
    """
    Stream `/v1/completions` and return (text, ttft_seconds, e2e_seconds).

    Notes:
    - TTFT is measured from request start until first non-[DONE] SSE chunk.
    - e2e is measured from request start until [DONE].
    """
    url = openai_base_url.rstrip("/") + "/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Authorization": "Bearer EMPTY"}

    start = timeit.default_timer()
    ttft = None
    text_parts: List[str] = []

    with requests.post(url, json=payload, headers=headers, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            if ttft is None:
                ttft = timeit.default_timer() - start
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            # OpenAI-style completions: choices[].text contains deltas for streaming
            choices = obj.get("choices", [])
            if choices:
                delta = choices[0].get("text")
                if isinstance(delta, str) and delta:
                    text_parts.append(delta)

    e2e = timeit.default_timer() - start
    if ttft is None:
        ttft = e2e
    return "".join(text_parts), float(ttft), float(e2e)

def _vllm_completion_non_stream_with_headers(
    *,
    openai_base_url: str,
    model: str,
    prompt: str,
    timeout_s: float = 180.0,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[str, Dict[str, str]]:
    """Call `/v1/completions` (non-streaming) and return (text, response_headers)."""
    url = openai_base_url.rstrip("/") + "/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Authorization": "Bearer EMPTY"}
    r = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    obj = r.json()
    text = ""
    choices = obj.get("choices", [])
    if choices and isinstance(choices[0], dict):
        t = choices[0].get("text")
        if isinstance(t, str):
            text = t
    return text, dict(r.headers)



# 1) Shared state schema
typedef = TypedDict  # compatibility alias
class GraphState(typedef('GraphState', {})):
    query: str
    urls: List[str]
    page_texts: List[str]
    summaries: List[str]
    final_response: str
    job_id: int
    skip_web_search: bool
 
# 2) Tool implementations with dynamic NVTX markers
 
def web_search(state: GraphState) -> GraphState:
    marker = f"web_search: {state['query'][:30]}"
    nvtx.push_range(marker)
    start_time = timeit.default_timer()

    if state['skip_web_search'] == False:
        api_key = os.getenv('TAVILY_API_KEY')
        if not api_key:
            nvtx.pop_range(); raise RuntimeError('Missing TAVILY_API_KEY. Set it at runtime or pass --skip-web-search.')
        client = TavilyClient(api_key=api_key)
        for attempt in range(3):
            try:
                response = client.search(state['query'], max_results=10)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise
        urls = [r['url'] for r in response.get('results', []) if 'url' in r]
    else:
        urls = ['https://en.wikipedia.org/wiki/Spiel_des_Jahres', 'https://boardgamegeek.com/wiki/page/Spiel_des_Jahres', 'https://www.reddit.com/r/boardgames/comments/buwap5/are_previous_spiel_des_jahres_winners_now_too/', 'https://boardgamegeek.com/thread/3282083/spiel-des-jahres-winners-1979-to-2023-and-who-do-y', 'https://blog.recommend.games/posts/thoughts-on-spiel-des-jahres/', 'https://www.spiel-des-jahres.de/en/award-winners-2024/', 'https://www.facebook.com/groups/132851767828/posts/10162746926537829/', 'https://www.tabletopgaming.co.uk/news/spiel-des-jahres-2024-winners-announced/', 'https://therewillbe.games/board-game-lists-and-guides/6214-the-ten-greatest-spiel-des-jahres-winners', 'https://www.dicebreaker.com/topics/spiel-des-jahres/best-games/overlooked-spiel-des-jahres-winners']
 

    elapsed = timeit.default_timer() - start_time
    timing_stats['web_search'].append(elapsed)
    with _trace_lock:
        idx = len(timing_stats['web_search']) - 1
        while len(query_traces) <= idx:
            query_traces.append({})
        query_traces[idx]['query_idx'] = idx
        query_traces[idx]['web_search'] = round(elapsed, 6)
    nvtx.pop_range()
    return {'urls': urls}
 
 
 
def _fetch_single(url: str, timeout: float = 10.0) -> Optional[str]:
    """Download one URL and return plain-text, or None on error."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser").get_text(separator="\n")
    except requests.RequestException:
        return None
 
 
 
def _fetch_url_single_state(state: GraphState) -> GraphState: 
    """Sequential download of up to two pages for *one* query."""
    marker = f"fetch_url: {state['query'][:30]}"
    nvtx.push_range(marker)
    start_time = timeit.default_timer()
 
    texts: List[str] = []
    for url in state["urls"]:
        if len(texts) >= 2:
            break
        txt = _fetch_single(url)
        if txt:
            texts.append(txt)
 
    elapsed = timeit.default_timer() - start_time
    timing_stats['fetch_url'].append(elapsed)
    with _trace_lock:
        idx = len(timing_stats['fetch_url']) - 1
        if idx < len(query_traces):
            query_traces[idx]['fetch_url'] = round(elapsed, 6)
    nvtx.pop_range()
    return {"page_texts": texts}
 
 
def fetch_url(state_or_states):  # LangGraph will pass list when batching
    """Batched wrapper: handles either a single state or a list of states."""
    if isinstance(state_or_states, list):
        # Parallelise across *queries* with a process pool
        max_workers = len(state_or_states)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_fetch_url_single_state, state_or_states))
        return results
    else:
        return _fetch_url_single_state(state_or_states)
 
 
# --- helper: picklable worker function ---
def _lexrank_one(text: str) -> str:
    """Run LexRank on a single document and return one-sentence summary."""
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    summarizer = LexRankSummarizer()
    doc = PlaintextParser.from_string(text, Tokenizer("english")).document
    sentences = summarizer(doc, sentences_count=1)
    return " ".join(str(s) for s in sentences)
 
# --- replace your current summarize() node ---
def summarize(state: GraphState) -> GraphState:
    marker = f"summarize: {state['query'][:30]}"
    nvtx.push_range(marker)
    start_time = timeit.default_timer() 
 
    if state["page_texts"]:
        max_workers = len(state["page_texts"])
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            sums = list(pool.map(_lexrank_one, state["page_texts"]))
    else:
        sums = []
 
 
    elapsed = timeit.default_timer() - start_time
    timing_stats['summarize'].append(elapsed)
    with _trace_lock:
        idx = len(timing_stats['summarize']) - 1
        if idx < len(query_traces):
            query_traces[idx]['summarize'] = round(elapsed, 6)
    nvtx.pop_range()
    return {"summaries": sums}

 
 
def final_answer(state: GraphState) -> GraphState:
    marker = f"llm_inference: {state['query'][:30]}"
    nvtx.push_range(marker)
    start_time = timeit.default_timer()
    openai_base_url = os.getenv("VLLM_OPENAI_BASE_URL", "http://localhost:5000/v1")
    model = os.getenv("VLLM_MODEL", "openai/gpt-oss-20b")
    prompt = f"Based on these summaries, answer: {state['query']}\n\n" + "\n\n".join(state['summaries'])

    # vLLM server now returns exact per-request timings as HTTP headers
    # (x-vllm-ttft-ms, x-vllm-e2e-ms, x-vllm-prefill-ms, x-vllm-decode-ms, ...).
    answer, resp_headers = _vllm_completion_non_stream_with_headers(
        openai_base_url=openai_base_url,
        model=model,
        prompt=prompt,
        max_tokens=int(os.getenv("VLLM_MAX_TOKENS", "256")),
        temperature=float(os.getenv("VLLM_TEMPERATURE", "0.0")),
    )

    def _header_ms(name: str) -> Optional[float]:
        v = resp_headers.get(name)
        if v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return None

    ttft_ms = _header_ms("x-vllm-ttft-ms")
    e2e_ms = _header_ms("x-vllm-e2e-ms")
    queue_ms = _header_ms("x-vllm-queue-ms")
    prefill_ms = _header_ms("x-vllm-prefill-ms")
    decode_ms = _header_ms("x-vllm-decode-ms")

    elapsed = timeit.default_timer() - start_time
    timing_stats['llm_inference'].append(elapsed)
    with _trace_lock:
        idx = len(timing_stats['llm_inference']) - 1
        if idx < len(query_traces):
            query_traces[idx]['llm_inference'] = round(elapsed, 6)
            # New: exact vLLM per-request timings (seconds), stored alongside
            # the old stages for notebook analysis.
            if ttft_ms is not None:
                query_traces[idx]['llm_ttft'] = round(ttft_ms / 1000.0, 6)
            if e2e_ms is not None:
                query_traces[idx]['llm_e2e'] = round(e2e_ms / 1000.0, 6)
            if queue_ms is not None:
                query_traces[idx]['llm_queue'] = round(queue_ms / 1000.0, 6)
            if prefill_ms is not None:
                query_traces[idx]['llm_prefill'] = round(prefill_ms / 1000.0, 6)
            if decode_ms is not None:
                query_traces[idx]['llm_decode'] = round(decode_ms / 1000.0, 6)
            query_traces[idx]['total'] = round(
                sum(query_traces[idx].get(s, 0) for s in ['web_search', 'fetch_url', 'summarize', 'llm_inference']), 6
            )
    nvtx.pop_range()
    return {'final_response': answer}
 
def print_timing_statistics():
    """Print average, min, and max time for each stage"""
    print("\n" + "="*70)
    print("TIMING STATISTICS (across all batches)")
    print("="*70)
    print(f"{'Stage':<20} {'Count':<10} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
    print("-"*70)
 
    for stage in ['web_search', 'fetch_url', 'summarize', 'llm_inference']:
        if stage in timing_stats and timing_stats[stage]:
            times = timing_stats[stage]
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            count = len(times)
            print(f"{stage:<20} {count:<10} {avg_time:<12.4f} {min_time:<12.4f} {max_time:<12.4f}")
        else:
            print(f"{stage:<20} {'0':<10} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
 
    print("="*70 + "\n")
 
# 3) Build and compile the graph
builder = StateGraph(GraphState)
builder.set_entry_point('web_search')
builder.add_node('web_search', web_search)
builder.add_node('fetch_url', fetch_url)
builder.add_node('summarize', summarize)
builder.add_node('final_answer', final_answer)
builder.add_edge('web_search', 'fetch_url')
builder.add_edge('fetch_url', 'summarize')
builder.add_edge('summarize', 'final_answer')
builder.set_finish_point('final_answer')
compiled_graph = builder.compile()
 
# 4) Batch invocation
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', action='store_true', help='Enable output of per-stage latencies')
    parser.add_argument('--skip-web-search', action='store_true', help='Skip web search stage, only for FreshQA benchmark')
    parser.add_argument('--sequential', action='store_true', help='Run multiple batches sequentially')
    parser.add_argument('--batch-size', type=int, default=1, help="Langchain batch size")
    parser.add_argument('--job-id', type=int, default=1, help="Job id for bash multiprocessing")
    parser.add_argument('--benchmark', choices=["freshQA", "musique", "QASC"], default="freshQA")
    parser.add_argument('--query-file', type=str, default=None, help="Path to a file with one question per line (overrides --benchmark)")
    parser.add_argument('--trace-output', type=str, default=None, help="Path to write per-query JSON traces")


    args = parser.parse_args()


    batch_size=args.batch_size
    if args.sequential:
        mini_batch = 1
    else:
        mini_batch = batch_size
    job_id = args.job_id

    if args.query_file:
        from pathlib import Path
        queries = [line.strip() for line in Path(args.query_file).read_text().splitlines() if line.strip()]
        if batch_size > len(queries):
            print(f"Warning: batch_size ({batch_size}) > available questions ({len(queries)}), capping to {len(queries)}")
            batch_size = len(queries)
        queries = queries[:batch_size]
    else:
        if args.benchmark == "freshQA":
            query_single = "Which game won the Spiel des Jahres award most recently?"
        elif args.benchmark == "musique":
            query_single = "When did the people who captured Malakoff come to the region where Philipsburg is located?"
        elif args.benchmark == "QASC":
            query_single = "Differential heating of air can be harnessed for what?"
        else:
            print("Wrong benchmark choice. Choose among 'freshQA', 'musique' and 'QASC'.")
            sys.exit()
        queries = [query_single] * batch_size
 
 
    initial_states = [
        {'query': q, 'urls': [], 'page_texts': [], 'summaries': [], 'final_response': '', 'job_id': job_id, 'skip_web_search': args.skip_web_search}
        for q in queries[0:batch_size]
    ]

    cfg = RunnableConfig(batch_size=batch_size, max_concurrency=mini_batch)
 
 
    nvtx.push_range('batch_run_all_queries')
    start_time = timeit.default_timer()
 
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {args.benchmark} | batch_size={batch_size}")
    print(f"{'='*70}")
    print(f"{job_id}: [TIMING] start: {start_time:.4f}s")

    result_states = compiled_graph.batch(initial_states, config=cfg)
    # print(result_state["final_response"])
    elapsed = timeit.default_timer() - start_time
    print(f"{job_id}: [TIMING] end: {elapsed:.4f}s")
    nvtx.pop_range()
 
    # Print timing statistics
    if args.verbose:
        print_timing_statistics()
 
    if args.trace_output:
        from pathlib import Path
        from datetime import datetime, timezone
        for i, trace in enumerate(query_traces):
            if i < len(result_states):
                trace['query'] = result_states[i].get('query', '')
                trace['answer'] = result_states[i].get('final_response', '')
                trace['summaries'] = result_states[i].get('summaries', [])
        trace_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark": args.query_file or args.benchmark,
            "batch_size": batch_size,
            "sequential": args.sequential,
            "job_id": job_id,
            "skip_web_search": args.skip_web_search,
            "total_wall_time": round(elapsed, 6),
            "metadata": _run_metadata(),
            "traces": query_traces,
        }
        Path(args.trace_output).write_text(json.dumps(trace_data, indent=2))
        print(f"Traces written to {args.trace_output} ({len(query_traces)} queries)")
 
 
