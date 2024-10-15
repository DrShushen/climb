import os
from typing import Any, Dict, List, cast

import pdfplumber
from openai import AzureOpenAI, OpenAI

from climb.common import Session
from climb.common.data_structures import UploadedFileAbstraction
from climb.engine.const import MODEL_MAX_MESSAGE_TOKENS

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase, UserInputRequest


# Initializing LLM for a sub-agent. TODO: Eventually factor this out of here.
def create_llm_client(
    session: Session,
    additional_kwargs_required: Dict[str, Any],
) -> Any:
    if session.engine_name in ("openai", "openai_nextgen"):
        client = OpenAI(api_key=additional_kwargs_required["api_key"])
    elif session.engine_name in ("azure_openai", "azure_openai_nextgen"):
        client = AzureOpenAI(
            azure_endpoint=additional_kwargs_required["azure_endpoint"],
            api_version=additional_kwargs_required["api_version"],
            api_key=additional_kwargs_required["api_key"],
        )
    else:
        raise ValueError(f"Unknown engine name: {session.engine_name}")
    return client


def get_llm_chat(
    client: Any,
    session: Session,
    additional_kwargs_required: Dict[str, Any],
    chat_kwargs: Dict,
) -> str:
    if session.engine_name in ("openai", "openai_nextgen"):
        model_type = additional_kwargs_required["engine_params"]["model_id"]
        out = client.chat.completions.create(
            model=model_type,
            max_tokens=MODEL_MAX_MESSAGE_TOKENS[model_type],
            temperature=additional_kwargs_required["engine_params"]["temperature"],
            # ---
            messages=chat_kwargs["messages"],
            stream=chat_kwargs["stream"],
        )
    elif session.engine_name in ("azure_openai", "azure_openai_nextgen"):
        model_type = additional_kwargs_required["azure_openai_config"].model
        out = client.chat.completions.create(
            model=additional_kwargs_required["azure_openai_config"].deployment_name,
            max_tokens=MODEL_MAX_MESSAGE_TOKENS[model_type],
            # ---
            messages=chat_kwargs["messages"],
            stream=chat_kwargs["stream"],
        )
    else:
        raise ValueError(f"Unknown engine name: {session.engine_name}")
    out_text = out.choices[0].message.content  # type: ignore
    return out_text


# --- --- --- --- ---


def upload_and_summarize_example_paper(
    tool_communicator: ToolCommunicator,
    paper_file_path: str,
    session: Session,
    additional_kwargs_required: Dict[str, Any],
) -> None:
    tool_communicator.print("Paper file uploaded successfully.")

    tool_communicator.print("Ingesting the paper PDF text...")
    paper_text = ""
    with pdfplumber.open(paper_file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            paper_text += f"\n\n{text}"

    tool_communicator.print("Summarizing the paper structure and style...")

    # NOTE: Left-aligned text below to avoid spurious spaces/tabs.
    SYSTEM_PROMPT = """
YOU ARE:
You are an expert in healthcare and medicine, with substantial knowledge of medical data analysis and writing medical papers.
You are reading a medical paper, which may be a clinical trial, a paper on a new predictive model, or any other medical paper.
You are tasked with summarizing the structure, flow, and style of the paper, but NOT THE CONTENT.
You are looking at this paper only to understand the structure and style of the writing in order to write a similar paper yourself based on your own data analysis.

YOUR TASK:
Please provide a clear, systematic report on the structure, flow, and style of the paper.
Be specific, and provide as much detail as possible.
Provide details about every section of the paper individually.

WHAT'S TRICKY:
You will be given a text representation of the paper, derived from the PDF.
This text is not perfect, and may contain errors, strange characters, or other issues.
Such problems will happen when converting a PDF to text, especially tables, figures, and other non-text elements.
Use your best understanding of PDF formatting to work around these issues.
"""
    FIRST_USER_MESSAGE = f"""
PAPER TEXT:

---
{paper_text}
---
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FIRST_USER_MESSAGE},
    ]

    client = create_llm_client(session=session, additional_kwargs_required=additional_kwargs_required)
    out_text = get_llm_chat(
        client=client,
        session=session,
        additional_kwargs_required=additional_kwargs_required,
        chat_kwargs={"messages": messages, "stream": False},
    )

    tool_communicator.set_returns(
        tool_return=f"Paper structure and style summary:\n\n{out_text}",
    )


class UploadAndSummarizeExamplePaper(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        # Handle the file upload.
        if self.user_input is None:
            raise ValueError("No user input obtained")
        if self.working_directory == "TO_BE_SET_BY_ENGINE":
            raise ValueError("Working directory not set")
        # for user_input in self.user_input:
        if self.user_input.kind == "file":
            uploaded_file = cast(UploadedFileAbstraction, self.user_input.received_input)
            paper_path = os.path.join(self.working_directory, uploaded_file.name)
            with open(paper_path, "wb") as f:
                f.write(uploaded_file.content)
        # else:
        #     raise NotImplementedError("Expected user input of kind 'file' but got something else.")

        thrd, out_stream = execute_tool(
            upload_and_summarize_example_paper,
            wd=self.working_directory,
            paper_file_path=paper_path,
            session=kwargs["session"],
            additional_kwargs_required=kwargs["additional_kwargs_required"],
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "upload_and_summarize_example_paper"

    @property
    def description(self) -> str:
        return """
        The user will be prompted to upload an example medical paper in PDF format via the UI.
        An assistant AI model will then be called to summarize the structure and style of the paper.
        This description will be added as an 'assistant' message for you to use. Note that the example paper may not \
        be a perfect match for our paper, so use this information as a guide.
        """

    @property
    def specification(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }

    @property
    def user_input_requested(self) -> List[UserInputRequest]:
        return [
            UserInputRequest(
                key="file", kind="file", description="Please upload your file", extra={"file_types": ["pdf"]}
            ),
        ]

    @property
    def description_for_user(self) -> str:
        return "allows you to upload an example paper in PDF format."
