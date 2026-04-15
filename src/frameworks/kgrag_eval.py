"""
KG-RAG Evaluator
================

Knowledge-Graph Based RAG System Evaluation Framework.

Reference: Dong et al., "Knowledge-Graph Based RAG System Evaluation Framework"
arXiv:2510.02549, October 2025

This framework extends RAGAS with knowledge graph-based evaluation paradigm,
enabling multi-hop reasoning and semantic community clustering for more
comprehensive scoring metrics.

Key Features:
- Multi-hop semantic matching via knowledge graph paths
- Semantic community clustering for factual accuracy
- More sensitive to subtle semantic differences than RAGAS
- Better correlation with human judgments on complex queries

Metrics:
- kg_faithfulness: KG-based faithfulness using entity alignment
- kg_relevancy: Multi-hop semantic relevancy scoring  
- community_overlap: Semantic community overlap between answer and context
- multi_hop_score: Multi-hop reasoning accuracy
"""

from typing import Dict, List, Set
import numpy as np
import re
from . import HeuristicEvaluator


class KGRAGEvaluator(HeuristicEvaluator):
    """
    KG-RAG: Knowledge-Graph Based RAG Evaluation (2025)
    
    Extends RAGAS with KG-based paradigm for:
    - Multi-hop reasoning evaluation
    - Semantic community clustering
    - Fine-grained factual accuracy assessment
    
    Paper findings show KG methods are more sensitive to semantic
    relevance/contrast than RAGAS, especially for entity-level evaluation.
    """
    
    def __init__(self, noise_std: float = 0.05):
        super().__init__(
            name="KG-RAG",
            metrics=["kg_faithfulness", "kg_relevancy", "community_overlap", "multi_hop_score"],
            noise_std=noise_std
        )
        self.version = "2025-10"  # October 2025
    
    def evaluate(self, sample: Dict) -> Dict[str, float]:
        """
        Evaluate using KG-RAG framework.
        
        KG-RAG decomposes text into atomic facts (entity-relation triples)
        and evaluates using knowledge graph alignment techniques.
        """
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        ground_truth = sample.get('ground_truth', {})
        
        # Extract entities and relations (simulated KG construction)
        answer_entities = self._extract_entities(answer)
        context_entities = self._extract_entities(context)
        question_entities = self._extract_entities(question)
        
        # KG-based faithfulness: entity alignment between answer and context
        kg_faithfulness = self._compute_kg_faithfulness(
            answer_entities, context_entities, answer, context
        )
        
        # Multi-hop semantic relevancy
        kg_relevancy = self._compute_kg_relevancy(
            question_entities, answer_entities, context_entities
        )
        
        # Semantic community overlap
        community_overlap = self._compute_community_overlap(
            answer, context, answer_entities, context_entities
        )
        
        # Multi-hop reasoning score
        multi_hop_score = self._compute_multi_hop_score(
            question, answer, context
        )
        
        return {
            'kg_faithfulness': self.add_noise(kg_faithfulness),
            'kg_relevancy': self.add_noise(kg_relevancy),
            'community_overlap': self.add_noise(community_overlap),
            'multi_hop_score': self.add_noise(multi_hop_score)
        }
    
    def _extract_entities(self, text: str) -> Set[str]:
        """
        Extract named entities from text (simplified NER).
        
        In real KG-RAG, this uses proper NER and entity linking.
        """
        if not text:
            return set()
        
        entities = set()
        
        # Capitalize words as candidate entities
        words = text.split()
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                entities.add(clean_word.lower())
        
        # Multi-word entities (consecutive capitalized)
        text_clean = re.sub(r'[^\w\s]', '', text)
        matches = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text_clean)
        for match in matches:
            entities.add(match.lower())
        
        # Numbers as potential entities (dates, quantities)
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        entities.update(numbers)
        
        return entities
    
    def _compute_kg_faithfulness(
        self, 
        answer_entities: Set[str], 
        context_entities: Set[str],
        answer: str,
        context: str
    ) -> float:
        """
        Compute KG-based faithfulness.
        
        KG-RAG computes entity alignment and checks if answer entities
        can be grounded in context via KG paths.
        """
        if not answer_entities:
            # Fallback to text-based if no entities
            return self.coverage_score(answer, context) if answer else 0.5
        
        # Entity overlap
        if context_entities:
            entity_overlap = len(answer_entities & context_entities) / len(answer_entities)
        else:
            entity_overlap = 0.0
        
        # Text-level grounding
        text_grounding = self.coverage_score(answer, context)
        
        # KG-RAG combines entity-level and text-level
        # Paper shows entity-level is more discriminative
        kg_faithfulness = 0.6 * entity_overlap + 0.4 * text_grounding
        
        return min(1.0, kg_faithfulness)
    
    def _compute_kg_relevancy(
        self,
        question_entities: Set[str],
        answer_entities: Set[str],
        context_entities: Set[str]
    ) -> float:
        """
        Compute multi-hop semantic relevancy.
        
        KG-RAG traces paths from question entities through context
        to answer entities (simulated).
        """
        if not question_entities or not answer_entities:
            return 0.5
        
        # Direct path: question -> answer
        direct_overlap = len(question_entities & answer_entities)
        
        # Bridge path: question -> context -> answer
        q_to_c = len(question_entities & context_entities) if context_entities else 0
        c_to_a = len(context_entities & answer_entities) if context_entities else 0
        
        # Multi-hop score
        direct_score = direct_overlap / len(question_entities) if direct_overlap else 0
        bridge_score = min(q_to_c, c_to_a) / len(question_entities) if q_to_c and c_to_a else 0
        
        # KG-RAG prefers grounded paths through context
        relevancy = 0.3 * direct_score + 0.7 * bridge_score
        
        return min(1.0, relevancy + 0.3)  # Base score boost
    
    def _compute_community_overlap(
        self,
        answer: str,
        context: str,
        answer_entities: Set[str],
        context_entities: Set[str]
    ) -> float:
        """
        Compute semantic community overlap.
        
        KG-RAG uses community detection on KG subgraphs.
        We simulate by clustering related terms.
        """
        if not answer or not context:
            return 0.0
        
        # Build word communities (simplified clustering)
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'between', 'under', 'again', 'further',
                     'then', 'once', 'here', 'there', 'when', 'where', 'why',
                     'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                     'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 'and', 'but', 'if', 'or', 'because',
                     'until', 'while', 'this', 'that', 'these', 'those'}
        
        answer_content = answer_words - stopwords
        context_content = context_words - stopwords
        
        if not answer_content:
            return 0.5
        
        # Community overlap: content words + entities
        content_overlap = len(answer_content & context_content) / len(answer_content)
        entity_overlap = len(answer_entities & context_entities) / len(answer_entities) if answer_entities else 0
        
        # Weighted combination (entities are higher signal)
        community_score = 0.4 * content_overlap + 0.6 * entity_overlap if answer_entities else content_overlap
        
        return min(1.0, community_score)
    
    def _compute_multi_hop_score(
        self,
        question: str,
        answer: str,
        context: str
    ) -> float:
        """
        Compute multi-hop reasoning score.
        
        KG-RAG evaluates if the answer requires information from
        multiple context segments (multi-hop reasoning).
        """
        if not question or not answer or not context:
            return 0.5
        
        # Split context into segments
        segments = [s.strip() for s in context.split('.') if len(s.strip()) > 20]
        
        if len(segments) < 2:
            # Single hop - just check basic grounding
            return self.coverage_score(answer, context)
        
        # Check if answer draws from multiple segments
        answer_words = set(answer.lower().split())
        segment_contributions = []
        
        for segment in segments:
            segment_words = set(segment.lower().split())
            contribution = len(answer_words & segment_words)
            segment_contributions.append(contribution)
        
        # Multi-hop indicator: answer uses multiple segments
        active_segments = sum(1 for c in segment_contributions if c > 2)
        multi_hop_ratio = active_segments / len(segments)
        
        # Also check if question entities are bridged through context
        question_coverage = self.word_overlap(question, context)
        answer_coverage = self.word_overlap(answer, context)
        
        # Multi-hop score: needs both question grounding and distributed answer
        multi_hop = 0.4 * multi_hop_ratio + 0.3 * question_coverage + 0.3 * answer_coverage
        
        return min(1.0, multi_hop)
