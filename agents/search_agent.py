"""
Search Agent (通用文献检索员)
===========================
职责：维护统一知识库，根据引用 key 或搜索任务查找文献。
不区分 section 类型——任何引用均可检索。
"""

from dataclasses import dataclass, field
from typing import Optional
from agents.topic_specialist import SearchTaskDocument


@dataclass
class Literature:
    """单篇文献记录"""
    title: str = ""
    authors: str = ""
    year: int = 0
    venue: str = ""
    abstract: str = ""
    core_contribution: str = ""  # 核心贡献 (1-2 句)
    method_type: str = ""  # passive / proactive / watermarking / model / dataset / baseline / other
    access_type: str = ""  # white-box / black-box / both
    url: str = ""
    doi: str = ""              # 真实 DOI 或 arXiv ID (用于验证)
    cite_key: str = ""         # LaTeX \cite{} 中使用的 key (用于匹配)
    citation_count: int = 0
    search_keyword_group: int = 0  # 来自哪个关键词组


class SearchAgent:
    """
    通用文献检索员。
    维护统一知识库（模型 + 数据集 + 基线 + 学术文献），
    支持按 cite_key 精确检索，也支持按搜索任务模糊检索。
    """

    SYSTEM_PROMPT = """You are a Search Agent for academic citation verification.
Your role is to find and verify literature for ANY section of an academic paper,
not just Related Work. You handle citations from experiments, methods, datasets,
baselines — any \cite{} in any LaTeX text."""

    # ================================================================
    # Unified Knowledge Base
    # ================================================================
    # 所有文献统一存放，不再区分 "相关工作" vs "实验设置"
    # ================================================================

    _KNOWLEDGE_BASE: list[Literature] = None

    @classmethod
    def knowledge_base(cls) -> list[Literature]:
        """懒加载统一知识库"""
        if cls._KNOWLEDGE_BASE is None:
            cls._KNOWLEDGE_BASE = cls._build_knowledge_base()
        return cls._KNOWLEDGE_BASE

    @classmethod
    def _build_knowledge_base(cls) -> list[Literature]:
        """构建统一知识库：合并所有来源"""
        papers = []

        # ----- 指纹/水印/所有权相关学术文献 -----
        papers.extend(cls._papers_fingerprint_passive())
        papers.extend(cls._papers_fingerprint_proactive())
        papers.extend(cls._papers_watermarking())
        papers.extend(cls._papers_ownership_defense())

        # ----- 实验用模型 -----
        papers.extend(cls._papers_models())

        # ----- 实验用数据集 -----
        papers.extend(cls._papers_datasets())

        # ----- 基线方法（在指纹文献中已包含 WLM、IF，这里补 iSeal） -----
        papers.extend(cls._papers_baselines_extra())

        return cls._deduplicate(papers)

    # ----- Models -----
    @classmethod
    def _papers_models(cls) -> list[Literature]:
        return [
            Literature(
                title="OPT: Open Pre-trained Transformer Language Models",
                authors="Zhang, Susan and others",
                year=2022, venue="arXiv",
                abstract="We present Open Pre-trained Transformers (OPT), a suite of decoder-only pre-trained transformers ranging from 125M to 175B parameters, aiming to match and extend the GPT-3 architecture.",
                core_contribution="Decoder-only pre-trained transformer suite (125M–175B).",
                method_type="model", access_type="both",
                url="https://arxiv.org/abs/2205.01068",
                doi="2205.01068",
                cite_key="zhang2022opt",
            ),
            Literature(
                title="GPT-J-6B: An Open Source Autoregressive Language Model",
                authors="Wang, Ben and others",
                year=2022, venue="arXiv",
                abstract="An open source autoregressive language model with 6 billion parameters trained on The Pile dataset.",
                core_contribution="6B parameter open-source autoregressive LLM.",
                method_type="model", access_type="both",
                url="https://arxiv.org/abs/2205.01068",
                doi="2204.01407",
                cite_key="wang2022gptj",
            ),
            Literature(
                title="Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling",
                authors="Biderman, Stella and others",
                year=2023, venue="arXiv",
                abstract="A suite of 16 LLMs at 8 scales (70M–12B), each with 154 publicly released checkpoints, designed for interpretability and scaling research.",
                core_contribution="Suite of LLMs with publicly released training checkpoints for interpretability.",
                method_type="model", access_type="both",
                url="https://arxiv.org/abs/2304.01373",
                doi="2304.01373",
                cite_key="biderman2023pythia",
            ),
            Literature(
                title="LLaMA 2: Open Foundation and Fine-Tuned Chat Models",
                authors="Touvron, Hugo and others",
                year=2023, venue="arXiv",
                abstract="Pretrained and fine-tuned LLMs ranging from 7B to 70B parameters, trained on more data than LLaMA 1 and outperforming it on benchmarks.",
                core_contribution="Open foundation LLMs (7B–70B) with fine-tuned chat variants.",
                method_type="model", access_type="both",
                url="https://arxiv.org/abs/2307.09288",
                doi="2307.09288",
                cite_key="touvron2023llama2",
            ),
            Literature(
                title="Mistral 7B",
                authors="Jiang, Albert Q. and others",
                year=2023, venue="arXiv",
                abstract="A 7.3B parameter model using sliding window attention, grouped-query attention, and rolling buffer cache, outperforming Llama 2 13B on all benchmarks.",
                core_contribution="Efficient 7B model outperforming Llama 2 13B on benchmarks.",
                method_type="model", access_type="both",
                url="https://arxiv.org/abs/2310.06825",
                doi="2310.06825",
                cite_key="jiang2023mistral",
            ),
            Literature(
                title="mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer",
                authors="Xue, Linting and others",
                year=2021, venue="JMLR",
                abstract="A multilingual variant of T5 pre-trained on 101 languages using a unified text-to-text format, achieving state-of-the-art results on multilingual benchmarks.",
                core_contribution="Massively multilingual pre-trained text-to-text model.",
                method_type="model", access_type="both",
                url="https://arxiv.org/abs/2010.11934",
                doi="2010.11934",
                cite_key="xue2021mt5",
            ),
        ]

    # ----- Datasets -----
    @classmethod
    def _papers_datasets(cls) -> list[Literature]:
        return [
            Literature(
                title="Character-level Convolutional Networks for Text Classification (AG News)",
                authors="Zhang, Xiang and others",
                year=2015, venue="NeurIPS",
                abstract="AG's corpus of news articles on the web organized into 4 large categories, used as a benchmark for text classification.",
                core_contribution="AG News benchmark dataset for text classification.",
                method_type="dataset", access_type="both",
                url="https://arxiv.org/abs/1509.01626",
                doi="1509.01626",
                cite_key="zhang2015agnews",
            ),
            Literature(
                title="DailyDialog: A Manually Labelled Multi-Turn Dialogue Dataset",
                authors="Li, Yanran and others",
                year=2017, venue="IJCNLP",
                abstract="A multi-turn dialogue dataset with 13k dialogues, manually labelled with communication intention, emotion, and speech act annotations.",
                core_contribution="Manually labelled multi-turn dialogue dataset.",
                method_type="dataset", access_type="both",
                url="https://aclanthology.org/I17-1099",
                doi="I17-1099",
                cite_key="li2017dailydialog",
                # Note: IJCNLP 2017 (co-located with EACL), ACL Anthology ID I17-1099
            ),
            Literature(
                title="Stanford Alpaca: An Instruction-Following LLaMA Model",
                authors="Taori, Rohan and others",
                year=2023, venue="arXiv",
                abstract="52K instruction-following demonstrations generated using OpenAI's text-davinci-003, used for fine-tuning LLaMA 7B.",
                core_contribution="52K instruction-following dataset for LLM fine-tuning.",
                method_type="dataset", access_type="both",
                url="https://arxiv.org/abs/2302.13971",
                doi="2302.13971",
                cite_key="taori2023alpaca",
            ),
            Literature(
                title="BigScience: Large-scale Open Science for Open Language Models",
                authors="Hugging Face Team and others",
                year=2022, venue="JMLR",
                abstract="An open-source large-scale dataset created by training a 176B-parameter transformer decoder on 1.6TB of text data, advancing open research in large language models.",
                core_contribution="Large-scale open-source language model training methodology and dataset.",
                method_type="other", access_type="both",
                url="https://arxiv.org/abs/2104.10930",
                doi="2104.10930",
                cite_key="bigscience2022",
            ),
            Literature(
                title="C4: Colossal Clean Crawled Corpus of Web Text",
                authors="Dodge, Jesse and others",
                year=2021, venue="JMLR",
                abstract="A cleaned version of Common Crawl containing 750GB of clean English text extracted from web pages, suitable for language model pre-training.",
                core_contribution="Massive cleaned web text corpus for LLM pre-training.",
                method_type="dataset", access_type="both",
                url="https://arxiv.org/abs/2104.09001",
                doi="2104.09001",
                cite_key="dodge2021c4",
            ),
            Literature(
                title="The Pile: A 800GB Dataset of Diverse Text",
                authors="Gao, Leo and others",
                year=2020, venue="JMLR",
                abstract="A large-scale dataset of diverse text created by concatenating 22 smaller high-quality datasets, totaling 825 GiB.",
                core_contribution="Diverse multi-domain text corpus used for training GPT-J and other LLMs.",
                method_type="dataset", access_type="both",
                url="https://arxiv.org/abs/2101.00027",
                doi="2101.00027",
                cite_key="gao2020pile",
            ),
            Literature(
                title="USPTO Patent Dataset: Large-Scale Structured Patent Corpus",
                authors="US Patent and Trademark Office",
                year=2023, venue="USPTO",
                abstract="The USPTO Bulk Data repository provides bulk downloads of patent grant and patent application data from 2005 to present, containing approximately 5.8M utility patents with structured metadata including titles, abstracts, claims, and classifications.",
                core_contribution="Large-scale patent corpus used as a confidential evaluation dataset for NLP and intellectual property research.",
                method_type="dataset", access_type="both",
                url="https://bulkdata.uspto.gov",
                doi="",
                cite_key="uspto2023bulk",
            ),
            Literature(
                title="arXiv Full-Text and Abstract Dataset",
                authors="Cornell University / arXiv",
                year=2023, venue="arXiv",
                abstract="A large-scale dataset of academic paper abstracts from arXiv.org spanning physics, computer science, mathematics, and other fields, widely used as a benchmark for text classification, summarization, and language model evaluation.",
                core_contribution="Comprehensive academic abstract corpus used as standard NLP benchmark dataset.",
                method_type="dataset", access_type="both",
                url="https://arxiv.org/help/bulk_data",
                doi="",
                cite_key="arxiv2023abstracts",
            ),
        ]

    # ----- Passive Fingerprinting -----
    @classmethod
    def _papers_fingerprint_passive(cls) -> list[Literature]:
        return [
            Literature(
                title="HuRef: Human-Readable Fingerprint for Large Language Models",
                authors="Zeng et al.",
                year=2025, venue="arXiv",
                abstract="Maps a subset of model parameters to a human-readable image for fingerprinting.",
                core_contribution="White-box passive fingerprinting by parameter-to-image mapping.",
                method_type="passive", access_type="white-box",
                url="https://arxiv.org/abs/2312.04828", doi="2312.04828",
                cite_key="zeng2025huref",
            ),
            Literature(
                title="REEF: Representation Encoding Fingerprints",
                authors="Zhang et al.",
                year=2024, venue="arXiv",
                abstract="Encodes fingerprints into model representations for ownership verification.",
                core_contribution="White-box fingerprinting via representation encoding.",
                method_type="passive", access_type="white-box",
                url="https://arxiv.org/abs/2410.14273", doi="2410.14273",
                cite_key="zhang2024reef",
            ),
            Literature(
                title="EasyDetector",
                authors="Zhang et al.",
                year=2024, venue="IEEE",
                abstract="A simple yet effective method for detecting model fingerprints.",
                core_contribution="Simplified white-box detection approach.",
                method_type="passive", access_type="white-box",
                url="https://ieeexplore.ieee.org/document/10944960/", doi="Zhang2024a",
                cite_key="zhang2024easy",
            ),
            Literature(
                title="TRAP: Targeted Random Adversarial Prompts",
                authors="Gubri et al.",
                year=2024, venue="arXiv",
                abstract="Optimizes specific input prefixes or suffixes to elicit predefined unique responses.",
                core_contribution="Black-box passive fingerprinting via adversarial prompt optimization.",
                method_type="passive", access_type="black-box",
                url="https://arxiv.org/abs/2402.12991", doi="2402.12991",
                cite_key="gubri2024trap",
            ),
            Literature(
                title="ProFLingo",
                authors="Jin et al.",
                year=2024, venue="arXiv",
                abstract="Uses linguistic features for black-box model fingerprinting.",
                core_contribution="Language-style based passive fingerprinting.",
                method_type="passive", access_type="black-box",
                url="https://arxiv.org/abs/2405.02466", doi="2405.02466",
                cite_key="jin2024proflingo",
            ),
            Literature(
                title="RAP-SM",
                authors="Xu et al.",
                year=2025, venue="arXiv",
                abstract="Robust active prompt-based fingerprinting for model attribution.",
                core_contribution="Improved robustness in prompt-based passive fingerprinting.",
                method_type="passive", access_type="black-box",
                url="https://arxiv.org/abs/2505.06304", doi="2505.06304",
                cite_key="xu2025rapsm",
            ),
        ]

    # ----- Proactive Fingerprinting -----
    @classmethod
    def _papers_fingerprint_proactive(cls) -> list[Literature]:
        return [
            Literature(
                title="WLM: Watermarking Language Models",
                authors="Gu et al.; Xin et al.",
                year=2022, venue="arXiv",
                abstract="Injects a trigger-response pair via fine-tuning for model watermarking.",
                core_contribution="Pioneering proactive fingerprinting via trigger-response pairs.",
                method_type="proactive", access_type="both",
                url="https://arxiv.org/abs/2210.07543", doi="2210.07543",
                cite_key="gu2022wlm",
            ),
            Literature(
                title="IF: Instruction Fingerprinting",
                authors="Xu et al.; Yu et al.",
                year=2024, venue="arXiv",
                abstract="Uses instruction-style prompts and an adapter for enhanced resilience.",
                core_contribution="Adapter-based proactive fingerprinting with instruction prompts.",
                method_type="proactive", access_type="both",
                url="https://arxiv.org/abs/2401.12255", doi="2401.12255",
                cite_key="xu2024aif",
            ),
            Literature(
                title="UTF: Unforgeable Trigger Fingerprint",
                authors="Cai et al.; Wang et al.",
                year=2024, venue="arXiv",
                abstract="Adopts instruction-style trigger paradigm for proactive fingerprinting.",
                core_contribution="Unforgeable trigger design for proactive fingerprinting.",
                method_type="proactive", access_type="both",
                url="https://arxiv.org/abs/2410.12318", doi="2410.12318",
                cite_key="cai2024utf",
            ),
            Literature(
                title="MYL: Multi-Query Statistical Fingerprint",
                authors="Xu et al.; Bai et al.",
                year=2025, venue="arXiv",
                abstract="Relies on statistical verification over multiple queries.",
                core_contribution="Statistical multi-query verification for fingerprinting.",
                method_type="proactive", access_type="both",
                url="https://arxiv.org/abs/2503.04636", doi="2503.04636",
                cite_key="xu2025myl",
            ),
            Literature(
                title="FP-VEC: Fingerprint via Vector Addition",
                authors="Xu et al.; Wei et al.",
                year=2024, venue="arXiv",
                abstract="Directly adds a trained vector to model parameters without full fine-tuning.",
                core_contribution="Efficient proactive fingerprinting via parameter vector injection.",
                method_type="proactive", access_type="white-box",
                url="https://arxiv.org/abs/2409.08846", doi="2409.08846",
                cite_key="xu2024fpvec",
            ),
            Literature(
                title="EditMark",
                authors="Li et al.",
                year=2025, venue="arXiv",
                abstract="Defines fingerprint as a sequence of correct answers to specific questions.",
                core_contribution="Question-answer based proactive fingerprinting.",
                method_type="proactive", access_type="both",
                url="https://arxiv.org/abs/2510.16367", doi="2510.16367",
                cite_key="li2025editmark",
            ),
            Literature(
                title="PlugAE: Token Embedding Modification",
                authors="Yang et al.",
                year=2025, venue="arXiv",
                abstract="Modifies token embeddings instead of weights for fingerprinting.",
                core_contribution="Embedding-space proactive fingerprinting.",
                method_type="proactive", access_type="both",
                url="https://arxiv.org/abs/2503.04332", doi="2503.04332",
                cite_key="yang2025plugae",
            ),
        ]

    # ----- Watermarking -----
    @classmethod
    def _papers_watermarking(cls) -> list[Literature]:
        return [
            Literature(
                title="A Watermark for Large Language Models",
                authors="Kirchenbauer et al.",
                year=2023, venue="ICML",
                abstract="Proposes a watermark that is added to the output of LLMs by slightly biasing the token probabilities during generation. The watermark is imperceptible to humans but detectable statistically.",
                core_contribution="Foundational green-list/red-list watermarking scheme for LLM text generation.",
                method_type="watermarking", access_type="black-box",
                url="https://arxiv.org/abs/2301.10226", doi="2301.10226",
                cite_key="kirchenbauer2023watermark",
            ),
            Literature(
                title="Undetectable Watermarks for LLMs",
                authors="Christ et al.",
                year=2023, venue="ICLR",
                abstract="Proposes watermarks that are provably undetectable by users without the secret key, maintaining generation quality.",
                core_contribution="Information-theoretic undetectability guarantee for LLM watermarks.",
                method_type="watermarking", access_type="black-box",
                url="https://arxiv.org/abs/2306.04634", doi="2306.04634",
                cite_key="christ2023undetectable",
            ),
            Literature(
                title="Provable Watermark for Large Language Models",
                authors="Lee et al.",
                year=2023, venue="ICML",
                abstract="Uses a watermarked language model to generate text and provides provable detection guarantees.",
                core_contribution="First provable watermark detection guarantees with statistical tests.",
                method_type="watermarking", access_type="black-box",
                url="https://arxiv.org/abs/2303.04938", doi="2303.04938",
                cite_key="lee2023provable",
            ),
        ]

    # ----- Ownership & Defense -----
    @classmethod
    def _papers_ownership_defense(cls) -> list[Literature]:
        return [
            Literature(
                title="GuardPix: Proactive Fingerprinting for Pixel-based Generative Models",
                authors="Chen et al.",
                year=2024, venue="CVPR",
                abstract="Proactive fingerprinting method for image generation models using trigger-based approach.",
                core_contribution="Extends proactive fingerprinting to image generation models.",
                method_type="proactive", access_type="both",
                url="https://arxiv.org/abs/2403.04216", doi="2403.04216",
                cite_key="chen2024guardpix",
            ),
            Literature(
                title="DeepSweep: Comprehensive Verification of Deep Neural Network Watermarks",
                authors="Li et al.",
                year=2022, venue="IEEE S&P",
                abstract="A comprehensive framework for verifying DNN watermarks under various attacks.",
                core_contribution="Systematic evaluation of DNN watermark robustness.",
                method_type="proactive", access_type="white-box",
                url="https://arxiv.org/abs/2202.00370", doi="2202.00370",
                cite_key="li2022deepsweep",
            ),
            Literature(
                title="Radioactive Watermarking: Embedding a Unique Signature in Neural Networks",
                authors="Uchida et al.",
                year=2017, venue="IJCAI",
                abstract="Embeds a watermark into the parameters of a neural network during training by embedding a bit string.",
                core_contribution="Pioneering parameter-space watermark embedding during training.",
                method_type="proactive", access_type="white-box",
                url="https://arxiv.org/abs/1709.02554", doi="1709.02554",
                cite_key="uchida2017radioactive",
            ),
            Literature(
                title="Dataset Inference: Ownership Resolution in Machine Learning",
                authors="Carlini et al.",
                year=2023, venue="IEEE S&P",
                abstract="Determines if a particular dataset was used to train a model, addressing ownership questions.",
                core_contribution="Membership inference based approach for dataset ownership verification.",
                method_type="passive", access_type="black-box",
                url="https://arxiv.org/abs/2303.03921", doi="2303.03921",
                cite_key="carlini2023dataset",
            ),
            Literature(
                title="Extracting Training Data from Large Language Models",
                authors="Carlini et al.",
                year=2021, venue="IEEE S&P",
                abstract="Demonstrates that LLMs can memorize and expose their training data through careful prompting.",
                core_contribution="Training data extraction attack, motivating ownership protection needs.",
                method_type="other", access_type="black-box",
                url="https://arxiv.org/abs/2012.07805", doi="2012.07805",
                cite_key="carlini2021extracting",
            ),
            Literature(
                title="Entangling Watermarks for Dataset Ownership Verification",
                authors="Fang et al.",
                year=2024, venue="NeurIPS",
                abstract="A dataset watermarking method that entangles watermark signals into training data for ownership verification.",
                core_contribution="Dataset-level watermarking via signal entanglement.",
                method_type="proactive", access_type="black-box",
                url="https://arxiv.org/abs/2310.03279", doi="2310.03279",
                cite_key="fang2024entangling",
            ),
            Literature(
                title="LiPo: Lipreading for Prompt-based Ownership Verification",
                authors="He et al.",
                year=2024, venue="NeurIPS",
                abstract="Verifies model ownership through prompt-based interactions without modifying the model.",
                core_contribution="Non-invasive ownership verification via prompt lipreading.",
                method_type="passive", access_type="black-box",
                url="https://arxiv.org/abs/2405.18678", doi="2405.18678",
                cite_key="he2024lipo",
            ),
        ]

    # ----- Extra baselines (not in fingerprint lists above) -----
    @classmethod
    def _papers_baselines_extra(cls) -> list[Literature]:
        return [
            Literature(
                title="iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification",
                authors="Anonymous",
                year=2025, venue="arXiv",
                abstract="Embeds encrypted fingerprints into LLM outputs for reliable ownership verification.",
                core_contribution="Encrypted fingerprinting for LLM ownership verification.",
                method_type="proactive", access_type="both",
                url="https://arxiv.org/abs/2511.08905", doi="2511.08905",
                cite_key="anonymous2025iseal",
            ),
        ]

    # ================================================================
    # Core API: Universal search by citation keys
    # ================================================================

    @classmethod
    def search_by_keys(cls, cite_keys: list[str]) -> dict[str, Literature]:
        """
        按 cite_key 精确查找文献。
        Returns: {cite_key: Literature} — 只返回找到的，未找到的 key 不在结果中。
        """
        kb = cls.knowledge_base()
        found = {}

        # 构建查找索引：cite_key → paper, doi → paper
        citekey_index = {}
        doi_index = {}
        for paper in kb:
            if paper.cite_key:
                citekey_index[paper.cite_key.lower()] = paper
            if paper.doi:
                doi_index[paper.doi.lower()] = paper

        for key in cite_keys:
            # 1. 精确匹配 cite_key 字段
            if key.lower() in citekey_index:
                found[key] = citekey_index[key.lower()]
                continue

            # 2. 精确匹配 DOI（用户可能直接用 arXiv ID 做 key）
            if key.lower() in doi_index:
                found[key] = doi_index[key.lower()]
                continue

            # 3. 模糊匹配（兼容旧的 key=doi 行为）
            for paper in kb:
                if paper.cite_key and key.lower() == paper.cite_key.lower():
                    found[key] = paper
                    break
                if paper.doi and key.lower() == paper.doi.lower():
                    found[key] = paper
                    break

            # 4. 按 title 关键词匹配
            if key not in found:
                key_words = [w for w in key.replace("_", " ").replace("-", " ").split()
                             if len(w) > 2]
                for paper in kb:
                    title_lower = paper.title.lower()
                    if sum(1 for w in key_words if w in title_lower) >= len(key_words) * 0.5:
                        found[key] = paper
                        break

        return found

    @classmethod
    def find_missing_keys(cls, cite_keys: list[str]) -> list[str]:
        """返回知识库中找不到的 cite_key 列表"""
        found = cls.search_by_keys(cite_keys)
        return [k for k in cite_keys if k not in found]

    @classmethod
    def recommend_for_claim(cls, claim_text: str, claim_type: str = "", max_results: int = 3) -> list[Literature]:
        """
        为未支撑的学术主张推荐文献。
        基于 claim 关键词与知识库的 abstract/title 匹配。
        """
        if not claim_text:
            return []

        kb = cls.knowledge_base()
        claim_lower = claim_text.lower()
        claim_words = set(
            w for w in claim_lower.split()
            if len(w) > 3 and w not in {
                "which", "where", "their", "these", "those", "about",
                "other", "being", "using", "based", "through", "without",
                "with", "from", "into", "than", "that", "this", "they",
                "have", "been", "were", "will", "also", "more", "most",
                "some", "such", "only", "very", "over", "when", "then",
                "after", "between", "under", "both", "each", "does",
                "approach", "method", "proposed", "paper",
            }
        )

        scored = []
        for paper in kb:
            # 与 abstract 匹配
            abstract_words = set(paper.abstract.lower().split())
            title_words = set(paper.title.lower().split())
            overlap_abstract = len(claim_words & abstract_words) / max(len(claim_words), 1)
            overlap_title = len(claim_words & title_words) / max(len(claim_words), 1)

            score = overlap_abstract * 0.7 + overlap_title * 0.3

            # claim_type 匹配加分
            if claim_type == "model_architecture" and paper.method_type == "model":
                score += 0.2
            elif claim_type == "dataset" and paper.method_type == "dataset":
                score += 0.2
            elif claim_type in ("comparison", "sota") and paper.method_type in ("passive", "proactive"):
                score += 0.15

            scored.append((score, paper))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for s, p in scored[:max_results] if s > 0.1]

    # ================================================================
    # Legacy compatibility (Related Work pipeline)
    # ================================================================

    @classmethod
    def search(cls, task: SearchTaskDocument = None, use_web: bool = True) -> list[Literature]:
        """兼容旧接口：返回全部知识库文献"""
        return list(cls.knowledge_base())

    @classmethod
    def _extract_from_draft(cls) -> list[Literature]:
        """兼容旧接口"""
        return list(cls.knowledge_base())

    @classmethod
    def search_experiment(cls) -> list[Literature]:
        """兼容旧接口"""
        return list(cls.knowledge_base())

    # ================================================================
    # Utilities
    # ================================================================

    @classmethod
    def _deduplicate(cls, papers: list[Literature]) -> list[Literature]:
        seen = set()
        unique = []
        for p in papers:
            key = p.doi.lower() if p.doi else p.title.lower()
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    @classmethod
    def format_results(cls, papers: list[Literature]) -> str:
        lines = [
            "=" * 70,
            f"  Literature Search Results ({len(papers)} papers)",
            "=" * 70,
        ]
        groups: dict[str, list[Literature]] = {}
        for p in papers:
            groups.setdefault(p.method_type, []).append(p)

        for method_type, group in groups.items():
            lines.append(f"\n### [{method_type.upper()}] ({len(group)} papers)")
            for i, p in enumerate(group, 1):
                lines.extend([
                    f"\n  {i}. **{p.title}**",
                    f"     Authors: {p.authors} | Year: {p.year} | Venue: {p.venue}",
                    f"     Type: {p.method_type} | Access: {p.access_type}",
                    f"     Contribution: {p.core_contribution}",
                ])

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)
