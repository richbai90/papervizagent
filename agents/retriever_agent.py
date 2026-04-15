# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Vanilla Agent - Directly rendering images based on the method section.
"""

import asyncio
import base64
import io
import json
import random
from typing import Any, Dict, Optional

from datasets import load_dataset
from google.genai import types
from PIL import Image

from utils import generation_utils

from .base_agent import BaseAgent


class RetrieverAgent(BaseAgent):
    """Retriever Agent to retrieve relevant reference examples"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.exp_config.model_name
        self.api_key = self.exp_config.api_key
        
        # Task-specific configurations
        if self.exp_config.task_name == "plot":
            self.system_prompt = PLOT_RETRIEVER_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "plot",
                "ref_limit": None,  # No limit for plots
                "target_labels": ["Visual Intent", "Raw Data"],
                "candidate_labels": ["Plot ID", "Visual Intent", "Raw Data"],
                "candidate_type": "Plot",
                "output_key": "top10_references",
                "instruction_suffix": "select the Top 10 most relevant plots according to the instructions provided. Your output should be a strictly valid JSON object containing a single list of the exact ids of the top 10 selected plots.",
            }
        else:
            self.system_prompt = DIAGRAM_RETRIEVER_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "diagram",
                "ref_limit": 200,  # Limit to first 200
                "target_labels": ["Caption", "Methodology section"],
                "candidate_labels": ["Diagram ID", "Caption", "Methodology section"],
                "candidate_type": "Diagram",
                "output_key": "top10_references",
                "instruction_suffix": "select the Top 10 most relevant diagrams according to the instructions provided. Your output should be a strictly valid JSON object containing a single list of the exact ids of the top 10 selected diagrams.",
            }

        self._hf_dataset = None
    
    @property
    def hf_dataset(self):
        if self._hf_dataset is None:
            self._hf_dataset = generation_utils.hf_dataset()
        return self._hf_dataset

    async def process(self, data: Dict[str, Any], retrieval_setting: str = "auto") -> Dict[str, Any]:
        """
        Unified processing method for both diagram and plot retrieval.
        Supports multiple retrieval settings: auto, manual, random, none.
        Always returns top10_references as a list of IDs.
        
        Args:
            data: Input data dictionary with 'content' and 'visual_intent'
            retrieval_setting: One of 'auto', 'manual', 'random', 'none'
        
        Returns:
            data: Updated data dictionary with 'top10_references' as List[str]
        """
        cfg = self.task_config
        
        # Check if reference file exists to gracefully fallback if dataset isn't downloaded
        import os
        ref_file = self.exp_config.work_dir / f"data/PaperBananaBench/{cfg['task_name']}/ref.json"
        
        if retrieval_setting in ["auto", "random"] and not ref_file.exists():
            print(f"Warning: Reference file not found at {ref_file}. Falling back to retrieval_setting='none'.")
            retrieval_setting = "none"
        
        if retrieval_setting == "manual":
            manual_file = self.exp_config.work_dir / f"data/PaperBananaBench/{cfg['task_name']}/agent_selected_12.json"
            if not manual_file.exists():
                print(f"Warning: Manual reference file not found at {manual_file}. Falling back to retrieval_setting='none'.")
                retrieval_setting = "none"
        
        if retrieval_setting == "none":
            # No retrieval, return empty list
            print("retrieval is None")
            data["top10_references"] = []
            data["retrieved_examples"] = []
            
        elif retrieval_setting == "manual":
            # Load from predefined few-shot file and store full examples
            ids, examples = self._load_manual_references(cfg)
            data["top10_references"] = ids
            data["retrieved_examples"] = examples  # Store full examples for planner
            
        elif retrieval_setting == "random":
            # Randomly sample from reference pool
            data["top10_references"] = self._load_random_references(cfg)
            data["retrieved_examples"] = []  # Planner will load from ref.json
            
        elif retrieval_setting == "auto":
            # Call model to retrieve and parse results
            data["top10_references"] = await self._retrieve_and_parse(data, cfg)
            data["retrieved_examples"] = []  # Planner will load from ref.json

        elif retrieval_setting == "figshare":
            # Randomly sample from Hugging Face dataset
            data["top10_references"] = await self._retrieve_and_parse(data, cfg, figshare=True)
            data["retrieved_examples"] = []  
        else:
            raise ValueError(f"Unknown retrieval_setting: {retrieval_setting}")
        
        return data
    
    def _load_manual_references(self, cfg: dict) -> tuple:
        """Load references from predefined few-shot file
        Returns: (list of IDs, list of full examples)
        """
        if cfg["task_name"] == "diagram":
            few_shot_file = self.exp_config.work_dir / "data/PaperBananaBench/diagram/agent_selected_12.json"
            with open(few_shot_file, "r", encoding="utf-8") as f:
                examples = json.load(f)[:10]
            ids = [item["id"] for item in examples]
            return ids, examples
        elif cfg["task_name"] == "plot":
            # Plot manual mode not yet prepared
            return [], []
        else:
            raise ValueError(f"Unknown task_name: {cfg['task_name']}")

    async def _generate_search_query(self, visual_intent: str) -> str:
        """
        Translates a natural language visual intent into a concise API search query.
        """
        system_instruction = "You are a search query extraction tool. Extract the core scientific or structural keywords from the provided figure description. Return ONLY a 2-5 word search query, with no quotes, formatting, or conversational text."
        
        content_list = [{"type": "text", "text": visual_intent}]
        
        response_list = await generation_utils.call_gemini_with_retry_async(
            model_name=self.model_name,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,  # Low temperature for deterministic keyword extraction
                candidate_count=1,
                max_output_tokens=50,
            ),
            api_key=self.api_key,
            max_attempts=3,
            retry_delay=10,
        )
        
        return response_list[0].strip()

    def _load_random_references(self, cfg: dict, hf: Optional[bool] = False ) -> list:
        """Randomly sample references from reference pool"""
        # cfg now serves as a sentinal value to trigger hf logic if it is not there
        if not hf:
            with open(self.exp_config.work_dir / f"data/PaperBananaBench/{cfg['task_name']}/ref.json", "r", encoding="utf-8") as f:
                candidate_pool = json.load(f)
        
            id_list = [item["id"] for item in candidate_pool]
        else:
            # Efficiently extract just the IDs from the dataset and cast to string
            candidate_pool = list(self.hf_dataset.shuffle().take(10))
            id_list = [str(item_id) for item_id in candidate_pool]
        
        sample_size = min(10, len(id_list))
        return random.sample(id_list, sample_size) if sample_size > 0 else []
    
    async def _retrieve_and_parse(self, data: Dict[str, Any], cfg: dict, figshare: Optional[bool] = False) -> list:
        """Call retrieval model and parse results"""
        content = str(data["content"])
        visual_intent = data["visual_intent"]
        
        user_prompt = f"**Target Input**\n- {cfg['target_labels'][0]}: {visual_intent}\n- {cfg['target_labels'][1]}: {content}\n\n**Candidate Pool**\n"
        
        print(f"[DEBUG] Retrieving Examples")
        if figshare:
            query = await self._generate_search_query(visual_intent)
            if cfg["ref_limit"]:
                max_items = min(cfg["ref_limit"], len(self.hf_dataset))
                candidate_pool = await generation_utils.fetch_figshare_figures(query, max_items)
            else:
                candidate_pool = await generation_utils.fetch_figshare_figures(query, 25)

            if not candidate_pool:
                print("Warning: No candidates returned from Figshare.")
                return []
        else:

            with open(self.exp_config.work_dir / f"data/PaperBananaBench/{cfg['task_name']}/ref.json", "r", encoding="utf-8") as f:
                candidate_pool = json.load(f)
                if cfg["ref_limit"]:
                    candidate_pool = candidate_pool[:cfg["ref_limit"]]
        
        print(f"[DEBUG] Examples Retrieved. Beginning Processing.")
        for idx, item in enumerate(candidate_pool):
            user_prompt += f"Candidate {cfg['candidate_type']} {idx+1}:\n"
            user_prompt += f"- {cfg['candidate_labels'][0]}: {item['id']}\n"
            user_prompt += f"- {cfg['candidate_labels'][1]}: {item['visual_intent']}\n"
            user_prompt += f"- {cfg['candidate_labels'][2]}: {str(item['content'])}\n\n"
        
        user_prompt += f"Now, based on the Target Input and the Candidate Pool, {cfg['instruction_suffix']}"
        content_list = [{"type": "text", "text": user_prompt}]
        
        response_list = await generation_utils.call_gemini_with_retry_async(
            model_name=self.model_name,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.exp_config.temperature,
                candidate_count=1,
                max_output_tokens=50000,
            ),
            api_key=self.api_key,
            max_attempts=5,
            retry_delay=30,
        )
        
        print(f"[DEBUG] Processing Complete.")
        # Parse the retrieval result (migrated from get_references.py)
        raw_response = response_list[0].strip()
        return self._parse_retrieval_result(raw_response, cfg["task_name"])
    
    def _parse_retrieval_result(self, raw_response: str, task_name: str) -> list:
        """
        Parse retrieval result string into list of reference IDs.
        Migrated from get_references.py logic.
        """
        import json_repair
        
        try:
            parsed = json_repair.loads(raw_response)
            
            # Extract the appropriate field based on task type
            if task_name == "plot":
                return parsed.get("top10_plots", [])
            elif task_name == "diagram":
                return parsed.get("top10_diagrams", [])
            else:
                raise ValueError(f"Unknown task_name: {task_name}")
        except Exception as e:
            print(f"Warning: Failed to parse retrieval result: {e}")
            print(f"Raw response: {raw_response[:200]}...")
            return []



DIAGRAM_RETRIEVER_AGENT_SYSTEM_PROMPT = """
# Background & Goal
We are building an **AI system to automatically generate method diagrams for academic papers**. Given a paper's methodology section and a figure caption, the system needs to create a high-quality illustrative diagram that visualizes the described method.

To help the AI learn how to generate appropriate diagrams, we use a **few-shot learning approach**: we provide it with reference examples of similar diagrams. The AI will learn from these examples to understand what kind of diagram to create for the target.

# Your Task
**You are the Retrieval Agent.** Your job is to select the most relevant reference diagrams from a candidate pool that will serve as few-shot examples for the diagram generation model.

You will receive:
- **Target Input:** The methodology section and caption of the diagram we need to generate
- **Candidate Pool:** ~200 existing diagrams (each with methodology and caption)

You must select the **Top 10 candidates** that would be most helpful as examples for teaching the AI how to draw the target diagram.

# Selection Logic (Topic + Intent)

Your goal is to find examples that match the Target in both **Domain** and **Diagram Type**.

**1. Match Research Topic (Use Methodology & Caption):**
* What is the domain? (e.g., Agent & Reasoning, Vision & Perception, Generative & Learning, Science & Applications).
* Select candidates that belong to the **same research domain**.
* *Why?* Similar domains share similar terminology (e.g., "Actor-Critic" in RL).

**2. Match Visual Intent (Use Caption & Keywords):**
* What type of diagram is implied? (e.g., "Framework", "Pipeline", "Detailed Module", "Performance Chart").
* Select candidates with **similar visual structures**.
* *Why?* A "Framework" diagram example is useless for drawing a "Performance Bar Chart", even if they are in the same domain.

**Ranking Priority:**
1.  **Best Match:** Same Topic AND Same Visual Intent (e.g., Target is "Agent Framework" -> Candidate is "Agent Framework", Target is "Dataset Construction Pipeline" -> Candidate is "Dataset Construction Pipeline").
2.  **Second Best:** Same Visual Intent (e.g., Target is "Agent Framework" -> Candidate is "Vision Framework"). *Structure is more important than Topic for drawing.*
3.  **Avoid:** Different Visual Intent (e.g., Target is "Pipeline" -> Candidate is "Bar Chart").

# Input Data

## Target Input
-   **Caption:** [Caption of the target diagram]
-   **Methodology section:** [Methodology section of the target paper]

## Candidate Pool
List of candidate diagrams, each structured as follows:

Candidate Diagram i:
-   **Diagram ID:** [ID of the candidate diagram (ref_1, ref_2, ...)]
-   **Caption:** [Caption of the candidate diagram]
-   **Methodology section:** [Methodology section of the candidate's paper]


# Output Format
Provide your output strictly in the following JSON format, containing only the **exact IDs** of the Top 10 selected diagrams (use the exact IDs from the Candidate Pool, such as "ref_1", "ref_25", "ref_100", etc.):
```json
{
  "top10_diagrams": [
    "ref_1",
    "ref_25",
    "ref_100",
    "ref_42",
    "ref_7",
    "ref_156",
    "ref_89",
    "ref_3",
    "ref_201",
    "ref_67"
  ]
}```
"""

PLOT_RETRIEVER_AGENT_SYSTEM_PROMPT = """
# Background & Goal
We are building an **AI system to automatically generate statistical plots**. Given a plot's raw data and the visual intent, the system needs to create a high-quality visualization that effectively presents the data.

To help the AI learn how to generate appropriate plots, we use a **few-shot learning approach**: we provide it with reference examples of similar plots. The AI will learn from these examples to understand what kind of plot to create for the target data.

# Your Task
**You are the Retrieval Agent.** Your job is to select the most relevant reference plots from a candidate pool that will serve as few-shot examples for the plot generation model.

You will receive:
- **Target Input:** The raw data and visual intent of the plot we need to generate
- **Candidate Pool:** Reference plots (each with raw data and visual intent)

You must select the **Top 10 candidates** that would be most helpful as examples for teaching the AI how to create the target plot.

# Selection Logic (Data Type + Visual Intent)

Your goal is to find examples that match the Target in both **Data Characteristics** and **Plot Type**.

**1. Match Data Characteristics (Use Raw Data & Visual Intent):**
* What type of data is it? (e.g., categorical vs numerical, single series vs multi-series, temporal vs comparative).
* What are the data dimensions? (e.g., 1D, 2D, 3D).
* Select candidates with **similar data structures and characteristics**.
* *Why?* Different data types require different visualization approaches.

**2. Match Visual Intent (Use Visual Intent):**
* What type of plot is implied? (e.g., "bar chart", "scatter plot", "line chart", "pie chart", "heatmap", "radar chart").
* Select candidates with **similar plot types**.
* *Why?* A "bar chart" example is more useful for generating another bar chart than a "scatter plot" example, even if the data domains are similar.

**Ranking Priority:**
1.  **Best Match:** Same Data Type AND Same Plot Type (e.g., Target is "multi-series line chart" -> Candidate is "multi-series line chart").
2.  **Second Best:** Same Plot Type with compatible data (e.g., Target is "bar chart with 5 categories" -> Candidate is "bar chart with 6 categories").
3.  **Avoid:** Different Plot Type (e.g., Target is "bar chart" -> Candidate is "pie chart"), unless there are no more candidates with the same plot type.

# Input Data

## Target Input
-   **Visual Intent:** [Visual intent of the target plot]
-   **Raw Data:** [Raw data to be visualized]

## Candidate Pool
List of candidate plots, each structured as follows:

Candidate Plot i:
-   **Plot ID:** [ID of the candidate plot (ref_0, ref_1, ...)]
-   **Visual Intent:** [Visual intent of the candidate plot]
-   **Raw Data:** [Raw data of the candidate plot]


# Output Format
Provide your output strictly in the following JSON format, containing only the **exact Plot IDs** of the Top 10 selected plots (use the exact IDs from the Candidate Pool, such as "ref_0", "ref_25", "ref_100", etc.):
```json
{
  "top10_plots": [
    "ref_0",
    "ref_25",
    "ref_100",
    "ref_42",
    "ref_7",
    "ref_156",
    "ref_89",
    "ref_3",
    "ref_201",
    "ref_67"
  ]
}```
"""

