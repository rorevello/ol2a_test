import argparse
import json
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr


CUSTOM_CSS = """
:root {
  --text: #202124;
  --muted: #5f6368;
  --line: #dadce0;
}

body, .gradio-container {
  background: #ffffff !important;
  color: var(--text);
}

.app-shell {
  max-width: 1240px;
  margin: 0 auto;
}

.hero {
  padding: 24px;
  border: 1px solid var(--line);
  background: #ffffff;
  border-radius: 12px;
}

.hero h1 {
  margin: 0 0 10px 0;
  font-size: 2.4rem;
  line-height: 1;
}

.hero p {
  margin: 0;
  color: var(--muted);
  font-size: 1.05rem;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-top: 18px;
}

.stat-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #ffffff;
  padding: 16px;
}

.stat-label {
  font-size: 0.84rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.stat-value {
  margin-top: 8px;
  font-size: 1.15rem;
  font-weight: 700;
}

.panel {
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #ffffff;
  box-shadow: none;
}

.panel-title {
  font-size: 1.15rem;
  font-weight: 700;
  margin-bottom: 6px;
}

.panel-subtitle {
  color: var(--muted);
  font-size: 0.95rem;
}

.status-banner {
  padding: 14px 16px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: #f8f9fa;
  color: var(--text);
  font-size: 0.95rem;
}

.warning-banner {
  background: #f8f9fa;
  color: var(--text);
}

.ok-banner {
  background: #f8f9fa;
  color: var(--text);
}

.answer-box {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px;
  background: #ffffff;
}

.answer-box h2, .answer-box h3 {
  margin-top: 0;
}

@media (max-width: 900px) {
  .stat-grid {
    grid-template-columns: 1fr;
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Gradio using the persisted Qwen/FAISS CORD-19 index."
    )
    parser.add_argument(
        "--qwen-index",
        type=Path,
        default=Path(__file__).resolve().parent / "hackathon" / "qwen.index",
        help="Persisted FAISS index. Its vectorizer must be at <index>.vect.",
    )
    parser.add_argument(
        "--dataset-name",
        default="cord19/fulltext/trec-covid",
        help="ir_datasets collection used to reconstruct indexed documents.",
    )
    parser.add_argument(
        "--ir-datasets-home",
        type=Path,
        default=Path(__file__).resolve().parent / ".ir_datasets",
        help="Local ir_datasets cache.",
    )
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Qwen model used when the CUDA pickle cannot be restored.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used to embed user queries.",
    )
    parser.add_argument(
        "--llm-base-url",
        default="http://192.168.212.254:8002/v1",
        help="Base URL for the OpenAI-compatible LLM server.",
    )
    parser.add_argument(
        "--llm-model",
        default="NousResearch/Hermes-4.3-36B",
        help="Remote model used to explain the retrieved evidence.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of chunks to retrieve for each query.",
    )
    parser.add_argument(
        "--server-name",
        default="0.0.0.0",
        help="Host for the Gradio server.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Preferred Gradio port. If busy, the app will try nearby ports.",
    )
    return parser.parse_args()


def load_store_if_available(
    index_path: Path,
    dataset_name: str,
    ir_datasets_home: Path,
    embedding_model: str,
    device: str,
) -> Tuple[Optional[Any], bool, str]:
    vectoriser_path = Path(f"{index_path}.vect")
    if not index_path.is_file():
        return (
            None,
            False,
            f"FAISS index not found at `{index_path}`.",
        )
    if not vectoriser_path.is_file():
        return (
            None,
            False,
            f"Qwen vectorizer not found at `{vectoriser_path}`.",
        )

    try:
        from hackathon.qwen_store import PersistentQwenStore

        store = PersistentQwenStore(
            index_path=index_path,
            dataset_name=dataset_name,
            ir_datasets_home=ir_datasets_home,
            model_name=embedding_model,
            device=device,
        )
    except Exception as exc:
        return (
            None,
            False,
            f"Could not load `{index_path}`. Error: `{exc}`",
        )

    return (
        store,
        True,
        (
            f"Loaded `{index_path}` with {store.size:,} CORD-19 documents. "
            f"Query vectorizer: {store.vectoriser_source}."
        ),
    )


def find_available_port(preferred_port: int, search_limit: int = 20) -> int:
    for port in range(preferred_port, preferred_port + search_limit + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise OSError(
        f"Could not find an open port between {preferred_port} and {preferred_port + search_limit}."
    )


def format_answer_html(title: str, body: str) -> str:
    return (
        "<div class='answer-box'>"
        f"<h2>{title}</h2>"
        f"{body}"
        "</div>"
    )


def summarize_results(results: List[Dict[str, object]]) -> str:
    lines = [
        "### Retrieved Evidence",
        f"Chunks retrieved: **{len(results)}**",
    ]
    for i, result in enumerate(results, start=1):
        lines.append(
            f"**{i}. {result['title']}**  \n"
            f"Section: `{result['section']}` | Score: `{result['score']:.4f}`  \n"
            f"{result['text'][:220]}..."
        )
    return "\n\n".join(lines)


def build_result_choices(results: List[Dict[str, object]]) -> List[str]:
    choices = []
    for i, result in enumerate(results, start=1):
        choices.append(
            f"{i}. {result['title']} | {result['section']} | score={result['score']:.4f}"
        )
    return choices


def get_selected_results(
    selected_labels: List[str],
    results: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    if not results:
        return []

    label_to_result = {
        label: result
        for label, result in zip(build_result_choices(results), results)
    }

    if not selected_labels:
        return results

    return [label_to_result[label] for label in selected_labels if label in label_to_result]


def render_selected_evidence(results: List[Dict[str, object]]) -> str:
    if not results:
        return "### Selected evidence\nNo retrieved documents are selected yet."

    blocks = ["### Selected evidence"]
    for i, result in enumerate(results, start=1):
        blocks.append(
            f"**{i}. {result['title']}**  \n"
            f"Section: `{result['section']}` | Score: `{result['score']:.4f}`  \n"
            f"{result['text'][:320]}..."
        )
    return "\n\n".join(blocks)


def render_follow_up_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "### Follow-up discussion\nAsk a question about the retrieved evidence to start the discussion."

    blocks = ["### Follow-up discussion"]
    for turn in history:
        blocks.append(f"**User:** {turn['question']}")
        blocks.append(f"**Assistant:** {turn['answer']}")
    return "\n\n".join(blocks)


def build_follow_up_prompt(
    question: str,
    history: List[Dict[str, str]],
) -> str:
    if not history:
        return question

    prior_turns = []
    for turn in history[-4:]:
        prior_turns.append(f"User: {turn['question']}")
        prior_turns.append(f"Assistant: {turn['answer']}")

    return (
        "Continue the conversation using only the selected retrieved evidence.\n\n"
        "Previous conversation:\n"
        + "\n".join(prior_turns)
        + f"\n\nNew user question:\n{question}"
    )


def main() -> None:
    args = parse_args()
    store, live_mode, launch_message = load_store_if_available(
        index_path=args.qwen_index,
        dataset_name=args.dataset_name,
        ir_datasets_home=args.ir_datasets_home,
        embedding_model=args.embedding_model,
        device=args.device,
    )
    banner_class = "ok-banner" if live_mode else "warning-banner"
    system_status = "Connected" if live_mode else "Waiting for index"

    def answer_question(prompt: str, top_k: int):
        if not prompt or not prompt.strip():
            return (
                format_answer_html(
                    "Missing query",
                    "<p>Please enter a question to start the retrieval flow.</p>",
                ),
                "### No results\nNo retrieval has been executed yet.",
                "[]",
                gr.update(choices=[], value=[]),
                [],
                "### Selected evidence\nNo retrieved documents are selected yet.",
                "### Follow-up discussion\nAsk a question about the retrieved evidence to start the discussion.",
                [],
            )

        if not live_mode or store is None:
            return (
                format_answer_html(
                    "Vector store unavailable",
                    (
                        "<p>The interface is ready, but the local index is not available yet.</p>"
                        f"<p>{launch_message}</p>"
                    ),
                ),
                "### No results\nThe vector store is not available.",
                "[]",
                gr.update(choices=[], value=[]),
                [],
                "### Selected evidence\nNo retrieved documents are selected yet.",
                "### Follow-up discussion\nAsk a question about the retrieved evidence to start the discussion.",
                [],
            )

        from rag_utils import query_hermes

        results = store.search(prompt, top_k=top_k)
        if not results:
            return (
                format_answer_html(
                    "No evidence found",
                    "<p>No chunks were retrieved for this query.</p>",
                ),
                "### No results\nThe retrieval step returned no chunks.",
                "[]",
                gr.update(choices=[], value=[]),
                [],
                "### Selected evidence\nNo retrieved documents are selected yet.",
                "### Follow-up discussion\nAsk a question about the retrieved evidence to start the discussion.",
                [],
            )

        selected_choices = build_result_choices(results)
        try:
            explanation = query_hermes(
                base_url=args.llm_base_url,
                model_name=args.llm_model,
                user_query=prompt,
                retrieval_results=results,
            )
            answer_title = "Hermes Explanation"
        except Exception as exc:
            answer_title = "Evidence retrieved"
            explanation = (
                "<p>The Qwen/FAISS search completed, but Hermes could not be reached.</p>"
                f"<p><code>{exc}</code></p>"
            )
        return (
            format_answer_html(answer_title, explanation),
            summarize_results(results),
            json.dumps(results, ensure_ascii=False, indent=2),
            gr.update(choices=selected_choices, value=selected_choices),
            results,
            render_selected_evidence(results),
            "### Follow-up discussion\nAsk a question about the retrieved evidence to start the discussion.",
            [],
        )

    def update_selected_preview(
        selected_labels: List[str],
        latest_results: List[Dict[str, object]],
    ):
        selected_results = get_selected_results(selected_labels or [], latest_results or [])
        return render_selected_evidence(selected_results)

    def ask_about_selected_docs(
        follow_up_question: str,
        selected_labels: List[str],
        latest_results: List[Dict[str, object]],
        history: List[Dict[str, str]],
    ):
        history = history or []
        latest_results = latest_results or []

        if not follow_up_question or not follow_up_question.strip():
            return (
                render_follow_up_history(history),
                history,
                "",
            )

        if not live_mode or store is None:
            notice = {
                "question": follow_up_question,
                "answer": "The vector store is not available yet, so follow-up questions cannot be answered.",
            }
            updated_history = history + [notice]
            return (
                render_follow_up_history(updated_history),
                updated_history,
                "",
            )

        selected_results = get_selected_results(selected_labels or [], latest_results)
        if not selected_results:
            notice = {
                "question": follow_up_question,
                "answer": "No retrieved documents are available for follow-up. Run a search first.",
            }
            updated_history = history + [notice]
            return (
                render_follow_up_history(updated_history),
                updated_history,
                "",
            )

        from rag_utils import query_hermes

        try:
            answer_text = query_hermes(
                base_url=args.llm_base_url,
                model_name=args.llm_model,
                user_query=build_follow_up_prompt(follow_up_question, history),
                retrieval_results=selected_results,
            )
        except Exception as exc:
            answer_text = f"Could not generate a follow-up answer. Error: {exc}"

        updated_history = history + [
            {
                "question": follow_up_question,
                "answer": answer_text,
            }
        ]
        return (
            render_follow_up_history(updated_history),
            updated_history,
            "",
        )

    examples = [
        ["What evidence is there about human infectivity from viral sequences?"],
        ["Summarize early studies on coronavirus transmission."],
        ["Which papers look most relevant to diagnosis and symptoms?"],
    ]

    with gr.Blocks(
        title="CORD-19 RAG with Qwen and Hermes",
        css=CUSTOM_CSS,
    ) as demo:
        latest_results_state = gr.State([])
        follow_up_history_state = gr.State([])

        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                f"""
                <section class="hero">
                  <h1>CORD-19 Retrieval Assistant</h1>
                  <p>
                    Search the literature with Qwen embeddings, retrieve evidence with FAISS,
                    and let Hermes explain the most relevant findings.
                  </p>
                  <div class="stat-grid">
                    <div class="stat-card">
                      <div class="stat-label">Index status</div>
                      <div class="stat-value">{system_status}</div>
                    </div>
                    <div class="stat-card">
                      <div class="stat-label">Embedding model</div>
                      <div class="stat-value">Qwen</div>
                    </div>
                    <div class="stat-card">
                      <div class="stat-label">Explanation model</div>
                      <div class="stat-value">Hermes 4.3 36B</div>
                    </div>
                  </div>
                </section>
                """
            )

            gr.HTML(
                f"<div class='status-banner {banner_class}'>{launch_message}</div>"
            )

            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown(
                            """
                            <div class="panel-title">1. User Query</div>
                            <div class="panel-subtitle">
                              Enter a question to run vector retrieval against the local index.
                            </div>
                            """
                        )
                        prompt = gr.Textbox(
                            label="Query",
                            placeholder="Example: What does the literature say about human infectivity from genomic sequences?",
                            lines=5,
                        )
                        top_k = gr.Slider(
                            label="Chunks to retrieve",
                            minimum=1,
                            maximum=8,
                            value=args.top_k,
                            step=1,
                        )
                        with gr.Row():
                            submit = gr.Button("Search", variant="primary")
                            clear = gr.Button("Clear")
                        gr.Examples(
                            examples=examples,
                            inputs=prompt,
                            label="Example queries",
                        )

                with gr.Column(scale=3):
                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown(
                            """
                            <div class="panel-title">2. Pipeline</div>
                            <div class="panel-subtitle">
                              End-to-end retrieval and explanation flow.
                            </div>
                            """
                        )
                        gr.Markdown(
                            """
                            - `Qwen` embeds the user query.
                            - `FAISS` searches the persisted `qwen.index`.
                            - `Hermes` generates the final explanation from retrieved evidence.
                            """
                        )

            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown(
                            """
                            <div class="panel-title">3. Final Answer</div>
                            <div class="panel-subtitle">
                              The model explanation will appear here.
                            </div>
                            """
                        )
                        answer = gr.HTML(
                            format_answer_html(
                                "Ready",
                                "<p>The interface is ready. Run a query once the vector store is available.</p>",
                            )
                        )

                with gr.Column(scale=4):
                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown(
                            """
                            <div class="panel-title">4. Retrieved Evidence</div>
                            <div class="panel-subtitle">
                              A quick summary of the top retrieved chunks.
                            </div>
                            """
                        )
                        retrieved_summary = gr.Markdown(
                            "### No results\nNo retrieval has been executed yet."
                        )

            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown(
                            """
                            <div class="panel-title">5. Focused Evidence Selection</div>
                            <div class="panel-subtitle">
                              Pick the retrieved documents you want to discuss in more detail.
                            </div>
                            """
                        )
                        selected_docs = gr.CheckboxGroup(
                            label="Retrieved documents",
                            choices=[],
                            value=[],
                        )
                        selected_preview = gr.Markdown(
                            "### Selected evidence\nNo retrieved documents are selected yet."
                        )

                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown(
                            """
                            <div class="panel-title">6. Follow-up Questions on Retrieved Documents</div>
                            <div class="panel-subtitle">
                              Ask the model about the specific retrieved evidence selected on the left.
                            </div>
                            """
                        )
                        follow_up_input = gr.Textbox(
                            label="Follow-up question",
                            placeholder="Example: Compare the first and second retrieved documents on transmission evidence.",
                            lines=3,
                        )
                        ask_follow_up = gr.Button("Ask about selected documents")
                        follow_up_transcript = gr.Markdown(
                            "### Follow-up discussion\nAsk a question about the retrieved evidence to start the discussion."
                        )

            with gr.Row():
                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown(
                        """
                        <div class="panel-title">7. Technical Output</div>
                        <div class="panel-subtitle">
                          Raw JSON for debugging and retrieval inspection.
                        </div>
                        """
                    )
                    retrieved_json = gr.Code(
                        value="[]",
                        label="Retrieved chunks",
                        language="json",
                    )

        submit.click(
            fn=answer_question,
            inputs=[prompt, top_k],
            outputs=[
                answer,
                retrieved_summary,
                retrieved_json,
                selected_docs,
                latest_results_state,
                selected_preview,
                follow_up_transcript,
                follow_up_history_state,
            ],
        )
        selected_docs.change(
            fn=update_selected_preview,
            inputs=[selected_docs, latest_results_state],
            outputs=[selected_preview],
        )
        ask_follow_up.click(
            fn=ask_about_selected_docs,
            inputs=[
                follow_up_input,
                selected_docs,
                latest_results_state,
                follow_up_history_state,
            ],
            outputs=[
                follow_up_transcript,
                follow_up_history_state,
                follow_up_input,
            ],
        )
        clear.click(
            fn=lambda: (
                "",
                4,
                format_answer_html(
                    "Ready",
                    "<p>The interface is ready. Run a query once the vector store is available.</p>",
                ),
                "### No results\nNo retrieval has been executed yet.",
                "[]",
                gr.update(choices=[], value=[]),
                [],
                "### Selected evidence\nNo retrieved documents are selected yet.",
                "### Follow-up discussion\nAsk a question about the retrieved evidence to start the discussion.",
                [],
            ),
            outputs=[
                prompt,
                top_k,
                answer,
                retrieved_summary,
                retrieved_json,
                selected_docs,
                latest_results_state,
                selected_preview,
                follow_up_transcript,
                follow_up_history_state,
            ],
        )

    launch_port = find_available_port(args.server_port)
    if launch_port != args.server_port:
        print(
            f"Port {args.server_port} is busy. Launching Gradio on port {launch_port} instead."
        )

    demo.launch(
        server_name=args.server_name,
        server_port=launch_port,
    )


if __name__ == "__main__":
    main()
