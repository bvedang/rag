"""
Adaptive RAG Chunker: BERTopic + ColBERT for Dynamic Topic Discovery
===================================================================
Production-ready chunking that automatically discovers topics without predefined keywords
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import os
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
import nltk

# Core models
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from rerankers import Reranker
from rerankers.documents import Document
from sentence_transformers import SentenceTransformer

# Clustering and metrics
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN
from umap import UMAP

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

@dataclass
class AdaptiveChunk:
    """Enhanced chunk with automatic topic discovery"""
    text: str
    sentences: List[str]
    sentence_indices: List[int]
    topic_id: Optional[int] = None
    topic_label: Optional[str] = None
    topic_keywords: List[str] = field(default_factory=list)
    topic_probability: float = 0.0
    coherence_score: float = 0.0
    embedding_centroid: Optional[np.ndarray] = None

@dataclass
class TopicInfo:
    """Information about a discovered topic"""
    topic_id: int
    label: str
    keywords: List[str]
    representative_docs: List[str]
    embedding: Optional[np.ndarray] = None
    document_count: int = 0
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())

class GlobalTopicRegistry:
    """Manages topics across multiple documents"""

    def __init__(self, registry_path: str = "topic_registry.json"):
        self.registry_path = registry_path
        self.topics: Dict[int, TopicInfo] = {}
        self.topic_embeddings: Dict[int, np.ndarray] = {}
        self.load_registry()

    def load_registry(self):
        """Load existing topic registry from disk"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    for topic_id, topic_data in data.items():
                        # Remove embedding from stored data as it's not JSON serializable
                        if 'embedding' in topic_data:
                            topic_data.pop('embedding')
                        self.topics[int(topic_id)] = TopicInfo(**topic_data)
            except Exception as e:
                print(f"Warning: Could not load registry: {e}")
                self.topics = {}

    def save_registry(self):
        """Save topic registry to disk"""
        data = {}
        for topic_id, topic_info in self.topics.items():
            data[str(topic_id)] = {
                'topic_id': topic_info.topic_id,
                'label': topic_info.label,
                'keywords': topic_info.keywords,
                'representative_docs': topic_info.representative_docs[:5],  # Keep top 5
                'document_count': topic_info.document_count,
                'created_date': topic_info.created_date
            }

        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_or_update_topic(self, topic_info: TopicInfo, embedding: np.ndarray):
        """Add new topic or update existing one"""
        if topic_info.topic_id in self.topics:
            # Update existing topic
            existing = self.topics[topic_info.topic_id]
            existing.document_count += 1
            existing.representative_docs.extend(topic_info.representative_docs)
            existing.representative_docs = list(set(existing.representative_docs))[:10]
        else:
            # Add new topic
            self.topics[topic_info.topic_id] = topic_info
            self.topic_embeddings[topic_info.topic_id] = embedding

    def find_similar_topic(self, embedding: np.ndarray, threshold: float = 0.85) -> Optional[int]:
        """Find existing topic similar to given embedding"""
        if not self.topic_embeddings:
            return None

        # Compare with existing topic embeddings
        similarities = []
        topic_ids = []

        for topic_id, topic_emb in self.topic_embeddings.items():
            sim = cosine_similarity([embedding], [topic_emb])[0][0]
            similarities.append(sim)
            topic_ids.append(topic_id)

        # Find best match
        if similarities:
            max_sim_idx = np.argmax(similarities)
            if similarities[max_sim_idx] > threshold:
                return topic_ids[max_sim_idx]

        return None

class AdaptiveRAGChunker:
    """
    Production-ready chunker that discovers topics automatically using BERTopic
    """

    def __init__(self,
                 embedding_model: str = "all-mpnet-base-v2",
                 colbert_model: str = "answerdotai/answerai-colbert-small-v1",
                 min_topic_size: int = 2,
                 colbert_weight: float = 0.4,
                 use_global_registry: bool = True,
                 registry_path: str = "topic_registry.json"):
        """
        Initialize adaptive chunker with BERTopic

        Args:
            embedding_model: Model for semantic embeddings
            colbert_model: ColBERT model for fine-grained similarity
            min_topic_size: Minimum sentences per topic
            colbert_weight: Weight for ColBERT vs topic embeddings
            use_global_registry: Whether to use global topic tracking
            registry_path: Path to save topic registry
        """
        print("🔄 Initializing Adaptive RAG Chunker...")

        # Initialize models
        self.sentence_model = SentenceTransformer(embedding_model)
        self.colbert_ranker = Reranker(colbert_model, model_type='colbert', verbose=0)

        # BERTopic configuration
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            min_topic_size=min_topic_size,
            n_gram_range=(1, 3),  # Consider phrases up to 3 words
            nr_topics="auto",  # Automatic topic detection
            calculate_probabilities=True,
            umap_model=UMAP(n_neighbors=15, n_components=5, random_state=42),
            hdbscan_model=HDBSCAN(min_cluster_size=min_topic_size, prediction_data=True),
            representation_model=KeyBERTInspired()  # Better topic representations
        )

        # Weights
        self.colbert_weight = colbert_weight
        self.topic_weight = 1 - colbert_weight

        # Global topic registry
        self.use_global_registry = use_global_registry
        if use_global_registry:
            self.topic_registry = GlobalTopicRegistry(registry_path)

        print("✅ Adaptive chunker initialized!")

    def _get_colbert_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get ColBERT embeddings"""
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))

        # Create documents
        docs = [Document(text=sent, doc_id=str(i)) for i, sent in enumerate(sentences)]

        # Calculate pairwise similarities
        for i, query_sent in enumerate(sentences):
            results = self.colbert_ranker.rank(query=query_sent, docs=docs)
            for result in results:
                j = int(result.doc_id)
                similarity_matrix[i, j] = result.score

        # Ensure symmetry
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
        np.fill_diagonal(similarity_matrix, 1.0)

        # Convert to embeddings using PCA
        pca = PCA(n_components=min(50, n-1))
        embeddings = pca.fit_transform(similarity_matrix)

        return embeddings

    def _discover_topics(self, sentences: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Discover topics using BERTopic

        Returns:
            topics: Topic assignment for each sentence
            probabilities: Topic probability for each sentence
            topic_info: Dictionary with topic information
        """
        # Get embeddings
        embeddings = self.sentence_model.encode(sentences, show_progress_bar=False)

        # Fit BERTopic
        topics, probabilities = self.topic_model.fit_transform(sentences, embeddings)

        # Ensure numpy arrays
        topics = np.array(topics)
        probabilities = np.array(probabilities)

        # Get topic information
        topic_info = {}
        for topic_id in set(topics):
            if topic_id != -1:  # Skip outlier topic
                # Get topic representation
                topic_words = self.topic_model.get_topic(topic_id)
                if topic_words:  # Check if topic words exist
                    keywords = [word for word, _ in topic_words[:10]]
                else:
                    keywords = []

                # Get representative documents
                topic_sentences = [sent for sent, t in zip(sentences, topics) if t == topic_id]

                # Create topic label from top keywords
                label = "_".join(keywords[:3]) if keywords else f"topic_{topic_id}"

                topic_info[topic_id] = {
                    'keywords': keywords,
                    'label': label,
                    'representative_docs': topic_sentences[:5],
                    'embedding': self._get_topic_embedding(topic_id, embeddings, topics)
                }

        return topics, probabilities, topic_info

    def _get_topic_embedding(self, topic_id: int, embeddings: np.ndarray,
                           topics: np.ndarray) -> np.ndarray:
        """Calculate centroid embedding for a topic"""
        # Ensure numpy array
        topics = np.array(topics)

        topic_mask = topics == topic_id
        if np.any(topic_mask):
            topic_embeddings = embeddings[topic_mask]
            return np.mean(topic_embeddings, axis=0)
        else:
            return np.zeros(embeddings.shape[1])

    def _combine_embeddings_with_topics(self, colbert_emb: np.ndarray,
                                      sentence_emb: np.ndarray,
                                      topic_assignments: np.ndarray,
                                      topic_probs: np.ndarray) -> np.ndarray:
        """
        Combine embeddings with discovered topic information
        """
        # Ensure numpy arrays
        topic_assignments = np.array(topic_assignments)
        if isinstance(topic_probs, list):
            topic_probs = np.array(topic_probs)

        # Normalize embeddings
        scaler = StandardScaler()
        colbert_scaled = scaler.fit_transform(colbert_emb)
        sentence_scaled = scaler.fit_transform(sentence_emb)

        # Create topic features
        # One-hot encode topics and weight by probability
        unique_topics = np.unique(topic_assignments[topic_assignments != -1])
        topic_features = np.zeros((len(topic_assignments), len(unique_topics)))

        for i, topic in enumerate(topic_assignments):
            if topic != -1 and len(unique_topics) > 0:
                # Find the index of this topic in unique_topics
                topic_indices = np.where(unique_topics == topic)[0]
                if len(topic_indices) > 0:
                    topic_idx = topic_indices[0]
                    # Handle both 1D and 2D probability arrays
                    if topic_probs.ndim == 1:
                        # 1D array: direct assignment
                        topic_features[i, topic_idx] = topic_probs[i]
                    else:
                        # 2D array: get probability for the assigned topic
                        # BERTopic returns probabilities for each topic
                        # We need to find which column corresponds to our topic
                        if hasattr(self.topic_model, 'topic_mapper_'):
                            # Map topic to column index
                            topic_col = self.topic_model._map_predictions([topic])[0]
                            if topic_col < topic_probs.shape[1]:
                                topic_features[i, topic_idx] = topic_probs[i, topic_col]
                        else:
                            # Fallback: use max probability
                            topic_features[i, topic_idx] = np.max(topic_probs[i])

        if topic_features.shape[1] > 0:
            topic_features_scaled = scaler.fit_transform(topic_features)
        else:
            topic_features_scaled = np.zeros((len(topic_assignments), 1))

        # Combine all features
        combined = np.concatenate([
            colbert_scaled * self.colbert_weight,
            sentence_scaled * self.topic_weight * 0.7,
            topic_features_scaled * self.topic_weight * 0.3
        ], axis=1)

        return combined

    def _cluster_with_topics(self, sentences: List[str],
                           combined_embeddings: np.ndarray,
                           topic_assignments: np.ndarray,
                           topic_probs: np.ndarray,
                           n_chunks: Optional[int] = None) -> List[AdaptiveChunk]:
        """
        Cluster sentences considering discovered topics
        """
        # Ensure numpy arrays
        topic_assignments = np.array(topic_assignments)
        if isinstance(topic_probs, list):
            topic_probs = np.array(topic_probs)

        # Determine optimal number of chunks
        if n_chunks is None:
            # Use number of discovered topics as starting point
            n_topics = len(np.unique(topic_assignments[topic_assignments != -1]))
            n_chunks = max(2, min(n_topics if n_topics > 0 else 3, 8))

            # Refine using silhouette score if we have enough sentences
            if len(sentences) > n_chunks + 1:
                best_score = -1
                best_k = n_chunks

                for k in range(max(2, n_topics-1), min(n_topics+3, len(sentences))):
                    try:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(combined_embeddings)
                        score = silhouette_score(combined_embeddings, labels)

                        if score > best_score:
                            best_score = score
                            best_k = k
                    except:
                        continue

                n_chunks = best_k

            print(f"📊 Optimal chunks determined: {n_chunks}")

        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_chunks,
            metric='euclidean',
            linkage='ward'
        )

        chunk_labels = clustering.fit_predict(combined_embeddings)

        # Create chunks
        chunks = []
        for chunk_id in range(n_chunks):
            # Get indices for this chunk
            chunk_indices = [i for i, label in enumerate(chunk_labels) if label == chunk_id]

            if not chunk_indices:
                continue

            # Get sentences for this chunk
            chunk_sentences = [sentences[i] for i in chunk_indices]

            # Determine dominant topic for chunk
            # Index with numpy array of indices
            chunk_topics = topic_assignments[np.array(chunk_indices)]
            if topic_probs.ndim == 1:
                chunk_probs = topic_probs[np.array(chunk_indices)]
            else:
                chunk_probs = topic_probs[np.array(chunk_indices)].max(axis=1)

            # Find most common topic (excluding outliers)
            valid_topics = chunk_topics[chunk_topics != -1]
            if len(valid_topics) > 0:
                # Use numpy's bincount to find most common topic
                topic_counts = np.bincount(valid_topics.astype(int))
                topic_id = np.argmax(topic_counts)
                # Get actual topic ID (in case there are gaps)
                unique_valid_topics = np.unique(valid_topics)
                if len(unique_valid_topics) > 0 and topic_id < len(unique_valid_topics):
                    topic_id = unique_valid_topics[min(topic_id, len(unique_valid_topics)-1)]
                    topic_prob = np.mean(chunk_probs[chunk_topics == topic_id])
                else:
                    topic_id = -1
                    topic_prob = 0.0
            else:
                topic_id = -1
                topic_prob = 0.0

            chunk = AdaptiveChunk(
                text=" ".join(chunk_sentences),
                sentences=chunk_sentences,
                sentence_indices=chunk_indices,
                topic_id=int(topic_id),
                topic_probability=float(topic_prob)
            )

            chunks.append(chunk)

        return chunks

    def _update_chunk_metadata(self, chunks: List[AdaptiveChunk],
                             topic_info: Dict,
                             embeddings: np.ndarray):
        """Update chunks with topic metadata and coherence scores"""
        for chunk in chunks:
            # Add topic information
            if chunk.topic_id != -1 and chunk.topic_id in topic_info:
                info = topic_info[chunk.topic_id]
                chunk.topic_label = info['label']
                chunk.topic_keywords = info['keywords']
            else:
                chunk.topic_label = "General"
                chunk.topic_keywords = []

            # Calculate coherence
            if len(chunk.sentence_indices) > 1:
                # Use numpy array for indexing
                chunk_embeddings = embeddings[np.array(chunk.sentence_indices)]
                similarities = cosine_similarity(chunk_embeddings)
                np.fill_diagonal(similarities, 0)
                n = len(chunk.sentence_indices)
                chunk.coherence_score = float(similarities.sum() / (n * (n - 1)))
            else:
                chunk.coherence_score = 1.0

            # Calculate embedding centroid
            chunk_embeddings = embeddings[np.array(chunk.sentence_indices)]
            chunk.embedding_centroid = np.mean(chunk_embeddings, axis=0)

    def chunk_document(self, text: str,
                      n_chunks: Optional[int] = None,
                      min_chunk_size: int = 2) -> Dict[str, Any]:
        """
        Main method to chunk document with automatic topic discovery

        Args:
            text: Document to chunk
            n_chunks: Number of chunks (if None, automatically determined)
            min_chunk_size: Minimum sentences per chunk

        Returns:
            Dictionary with chunks and analysis
        """
        # Split into sentences
        sentences = sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        print(f"\n📄 Processing document with {len(sentences)} sentences...")

        if len(sentences) < min_chunk_size:
            return {
                'chunks': [AdaptiveChunk(
                    text=text,
                    sentences=sentences,
                    sentence_indices=list(range(len(sentences)))
                )],
                'topics_discovered': 0,
                'analysis': {}
            }

        # Step 1: Discover topics
        print("🔍 Discovering topics with BERTopic...")
        topic_assignments, topic_probs, topic_info = self._discover_topics(sentences)
        n_topics = len([t for t in set(topic_assignments) if t != -1])
        print(f"   Found {n_topics} topics")

        # Step 2: Get embeddings
        print("🧮 Generating embeddings...")
        colbert_embeddings = self._get_colbert_embeddings(sentences)
        sentence_embeddings = self.sentence_model.encode(sentences, show_progress_bar=False)

        # Step 3: Combine embeddings with topics
        print("🔀 Combining embeddings with discovered topics...")
        combined_embeddings = self._combine_embeddings_with_topics(
            colbert_embeddings, sentence_embeddings,
            topic_assignments, topic_probs
        )

        # Step 4: Cluster with topic awareness
        print("📦 Creating semantic chunks...")
        chunks = self._cluster_with_topics(
            sentences, combined_embeddings,
            topic_assignments, topic_probs, n_chunks
        )

        # Step 5: Update chunk metadata
        self._update_chunk_metadata(chunks, topic_info, sentence_embeddings)

        # Step 6: Update global registry if enabled
        if self.use_global_registry and topic_info:
            self._update_global_registry(topic_info, topic_assignments)

        # Step 7: Create analysis
        analysis = self._analyze_results(
            sentences, chunks, topic_info,
            topic_assignments, combined_embeddings
        )

        return {
            'chunks': chunks,
            'topics_discovered': n_topics,
            'topic_info': topic_info,
            'analysis': analysis,
            'embeddings': {
                'colbert': colbert_embeddings,
                'sentence': sentence_embeddings,
                'combined': combined_embeddings
            }
        }

    def _update_global_registry(self, topic_info: Dict, topic_assignments: np.ndarray):
        """Update global topic registry with new topics"""
        # Ensure numpy array
        topic_assignments = np.array(topic_assignments)

        for topic_id, info in topic_info.items():
            # Check if similar topic exists
            similar_topic_id = self.topic_registry.find_similar_topic(
                info['embedding'], threshold=0.85
            )

            if similar_topic_id is not None:
                # Merge with existing topic
                print(f"   Merging with existing topic: {self.topic_registry.topics[similar_topic_id].label}")
                topic_assignments[topic_assignments == topic_id] = similar_topic_id
            else:
                # Add as new topic
                new_topic = TopicInfo(
                    topic_id=len(self.topic_registry.topics),
                    label=info['label'],
                    keywords=info['keywords'],
                    representative_docs=info['representative_docs'],
                    document_count=1
                )
                self.topic_registry.add_or_update_topic(new_topic, info['embedding'])
                print(f"   Added new topic to registry: {info['label']}")

        self.topic_registry.save_registry()

    def _analyze_results(self, sentences: List[str], chunks: List[AdaptiveChunk],
                        topic_info: Dict, topic_assignments: np.ndarray,
                        embeddings: np.ndarray) -> Dict:
        """Analyze chunking results"""
        # Ensure numpy array
        topic_assignments = np.array(topic_assignments)

        analysis = {
            'n_sentences': len(sentences),
            'n_chunks': len(chunks),
            'n_topics': len(topic_info),
            'avg_chunk_size': np.mean([len(c.sentences) for c in chunks]) if chunks else 0,
            'avg_coherence': np.mean([c.coherence_score for c in chunks]) if chunks else 0,
            'topic_distribution': {}
        }

        # Topic distribution
        for chunk in chunks:
            if chunk.topic_label:
                if chunk.topic_label not in analysis['topic_distribution']:
                    analysis['topic_distribution'][chunk.topic_label] = 0
                analysis['topic_distribution'][chunk.topic_label] += len(chunk.sentences)

        # Topic purity (how well chunks align with topics)
        topic_purity_scores = []
        for chunk in chunks:
            if chunk.topic_id != -1 and len(chunk.sentence_indices) > 0:
                # Use numpy array for indexing
                chunk_topics = topic_assignments[np.array(chunk.sentence_indices)]
                valid_topics = chunk_topics[chunk_topics != -1]
                if len(valid_topics) > 0:
                    purity = np.mean(valid_topics == chunk.topic_id)
                    topic_purity_scores.append(purity)

        analysis['avg_topic_purity'] = float(np.mean(topic_purity_scores)) if topic_purity_scores else 0.0

        return analysis

    def visualize_results(self, result: Dict, save_path: Optional[str] = None):
        """Visualize chunking results with discovered topics"""
        chunks = result['chunks']
        analysis = result['analysis']
        topic_info = result.get('topic_info', {})

        fig = plt.figure(figsize=(20, 12))

        # 1. Topic discovery visualization
        ax1 = plt.subplot(2, 3, 1)
        if hasattr(self.topic_model, 'topic_embeddings_') and self.topic_model.topic_embeddings_ is not None:
            try:
                # Use BERTopic's internal visualization
                topic_embeddings_2d = UMAP(n_neighbors=15, n_components=2,
                                          random_state=42).fit_transform(self.topic_model.topic_embeddings_)

                for i, topic_id in enumerate(topic_info):
                    if topic_id != -1 and i < len(topic_embeddings_2d):
                        ax1.scatter(topic_embeddings_2d[i, 0],
                                  topic_embeddings_2d[i, 1],
                                  s=200, alpha=0.7,
                                  label=f"Topic {topic_id}: {topic_info[topic_id]['label'][:20]}...")

                ax1.set_title('Discovered Topics in 2D Space')
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            except Exception as e:
                ax1.text(0.5, 0.5, f"Topic visualization not available: {e}",
                        transform=ax1.transAxes, ha='center')
        else:
            ax1.text(0.5, 0.5, "Topic embeddings not available",
                    transform=ax1.transAxes, ha='center')

        # 2. Chunk distribution
        ax2 = plt.subplot(2, 3, 2)
        colors = plt.cm.tab20(np.linspace(0, 1, len(chunks)))

        for i, chunk in enumerate(chunks):
            positions = chunk.sentence_indices
            ax2.scatter(positions, [i] * len(positions),
                       c=[colors[i]], s=150, alpha=0.7,
                       label=f'{chunk.topic_label[:15]} (coh: {chunk.coherence_score:.3f})')

        ax2.set_xlabel('Original Sentence Position')
        ax2.set_ylabel('Chunk ID')
        ax2.set_title('Sentence Distribution Across Chunks')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        # 3. Topic keywords word cloud style
        ax3 = plt.subplot(2, 3, 3)
        ax3.axis('off')
        y_pos = 0.9

        for topic_id, info in list(topic_info.items())[:5]:  # Top 5 topics
            keywords_str = ", ".join(info['keywords'][:5])
            ax3.text(0.1, y_pos, f"Topic {topic_id}: {keywords_str}",
                    fontsize=10, transform=ax3.transAxes)
            y_pos -= 0.15

        ax3.set_title('Discovered Topic Keywords')

        # 4. Chunk coherence
        ax4 = plt.subplot(2, 3, 4)
        coherence_scores = [c.coherence_score for c in chunks]
        if coherence_scores:
            bars = ax4.bar(range(len(chunks)), coherence_scores, color=colors)
            ax4.set_xlabel('Chunk')
            ax4.set_ylabel('Coherence Score')
            ax4.set_title('Chunk Coherence Scores')
            ax4.set_xticks(range(len(chunks)))
            ax4.set_xticklabels([c.topic_label[:10] for c in chunks], rotation=45, ha='right')
            ax4.axhline(y=np.mean(coherence_scores), color='red', linestyle='--',
                       label=f'Average: {np.mean(coherence_scores):.3f}')
            ax4.legend()

        # 5. Topic distribution
        ax5 = plt.subplot(2, 3, 5)
        if analysis['topic_distribution']:
            labels = list(analysis['topic_distribution'].keys())
            sizes = list(analysis['topic_distribution'].values())
            ax5.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)])
            ax5.set_title('Overall Topic Distribution')

        # 6. Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        summary_text = f"""Summary Statistics:

Total Sentences: {analysis['n_sentences']}
Number of Chunks: {analysis['n_chunks']}
Topics Discovered: {analysis['n_topics']}
Avg Chunk Size: {analysis['avg_chunk_size']:.1f}
Avg Coherence: {analysis['avg_coherence']:.3f}
Avg Topic Purity: {analysis.get('avg_topic_purity', 0):.3f}

Top Topics:
"""
        for topic, count in list(analysis.get('topic_distribution', {}).items())[:5]:
            summary_text += f"\n  • {topic}: {count} sentences"

        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


def demonstrate_adaptive_chunking():
    """Demonstrate adaptive chunking with automatic topic discovery"""

    print("🚀 Adaptive RAG Chunking with BERTopic")
    print("=" * 60)
    print("\n📝 Features:")
    print("   • Automatic topic discovery - no predefined keywords!")
    print("   • Global topic registry for consistency")
    print("   • Works with any document type")
    print("   • Production-ready for RAG pipelines\n")

    # Test with the same document
    document = """
    VEDANG BARHATE
    Bellevue, WA, 98007 | +1 (425) 518-3273 | vedang.inbox@gmail.com | https://www.linkedin.com/in/vedangbarhate SKILLS AND CERTIFICATIONS
    Programming Languages: Java, Python, JavaScript, TypeScript, C++, SQL
    Development & Testing: React, Next.js, Vite, Tailwind CSS, Spring Boot, Hibernate, FastAPI, JUnit, Mockito, Pytest, Jest
    Data & Infrastructure: PostgreSQL, DynamoDB, Flyway, AWS, S3, Docker, Jenkins, Git, GitHub, Bitbucket, Postman, X-Ray
    Certifications: AWS Developer, AWS Solutions Architect, AWS Cloud Practitioner, Deep Learning Specialization.
    EXPERIENCE
    Software Development Engineer | FIS, Bellevue, US June 2024 - Present
    Project: - Case-whisperer | (Next.js, FastAPI, DynamoDB, Kafka, EKS)
    ● Architected greenfield AI case platform from ground up, implementing Google's A2A and MCP protocols for agent communication, reducing mean time to resolve cases from 35 to 18 minutes across fraud and compliance workflows
    ● Built distributed agent orchestration on EKS processing 1000+ cases/hour using Azure OpenAI GPT-4o-mini, with Kafka/MSK architecture supporting configurable JSON-based workflows and horizontal scaling to 5000+ cases/hour during peak loads
    ● Built automation coverage tracking system measuring AI vs manual investigation ratios, enabling elastic scaling to handle 720K+ monthly cases without proportional increase in analyst headcount
    ● Designed configurable agent workflows for fraud detection, compliance analysis, and case management with automated routing through summarization and false-positive detection stages, achieving 80% automation coverage
    Project: - Transaction Fraud Management | (Next.js, Spring Boot, PostgreSQL, Dynamodb, Kinesis, DMS, S3, EKS)
    ● Bootstrapped greenfield CaseManager microservice from zero codebase, establishing Spring Boot architecture with modular service layers, RESTful APIs, and testing framework, delivering production-ready MVP in 8 weeks supporting initial 50K+ cases/month
    ● Built stream processing system handling 450M+ monthly financial transactions across 5+ tenants (3.5M cases/tenant), implementing distributed Kinesis workers achieving 1000 TPS per instance with horizontal scaling to 2500+ TPS during peak periods
    ● Built PostgreSQL CDC-to-Kinesis pipeline processing 91M+ monthly events per tenant with zero data loss, utilizing KCL batch processing (100 records/batch), envelope encryption with key caching, and optimized checkpointing to maintain sub-second latency
    ● Led migration of 3 core services (CaseService, AttachmentService, AlertService) from DynamoDB to PostgreSQL, redesigning data models for relational paradigm and implementing JOIN operations across case relationships, alerts, and transaction data
    ● Designed multi-layered encryption architecture for financial case management using AWS KMS envelope encryption, optimizing performance by 60% through unified encryption context strategy that reduced P95 latency from 500 ms to 200 ms
    ● Implemented auto-scaling stream workers handling 4x traffic variations (128-512 TPS/tenant), with automatic shard rebalancing across 5 instances and sub-15 second failover recovery, ensuring 99.9% uptime for PCI-compliant financial transactions
    ● Reduced deployment friction through custom CI/CD automation in sandbox-constrained environment, architecting ECR/ECS/ALB infrastructure and shell scripts that transformed manual 30-min deployments into 2-minute automated pipelines for 10+ developers
    ● Built comprehensive runbooks covering on-call procedures and debugging workflows as we expanded from sandbox to dev/UAT/prod environments, enabling team members to independently handle deployment failures, stream worker issues, and database problems
    Software Engineer | Beyond Key Solution, IL Feb 2023 – May 2024
    Projects: Document Orchestration & Supplier Management Platform | (Spring Boot, React, AWS, Elasticsearch, D3.js)
    ● Built cloud-based Document Orchestration Service processing 4000+ daily uploads, implementing automated workflow with Apache PDFBox that reduced processing time by 25% through parallel processing and batch optimization
    ● Developed adaptive throttling mechanism for SQS message processing, dynamically adjusting dequeue rates based on CloudWatch metrics from Elasticsearch cluster, preventing overload during 10x traffic spikes without infrastructure scaling
    ● Optimized Elasticsearch search performance through custom indexing strategy and query optimization, reducing search response time from 2s to 200ms across 1M+ document corpus
    ● Engineered supplier dissociation platform streamlining workflows for 200+ users, reducing manual processing from 45 to 15 minutes per transition while building real-time dashboard with D3.js/Recharts for 15+ KPIs
    ● Implemented comprehensive testing strategy achieving 85%+ code coverage across Spring Boot services and React components, reducing production bugs by 70%
    PROJECTS
    WAL Kinesis Streamer | (Python, PostgreSQL, LocalStack Kinesis, Asyncio)
    ● Built local PostgreSQL CDC alternative to AWS DMS, enabling developers to test streaming pipelines without cloud dependencies and reducing integration testing feedback loop from 20 minutes to 2 minutes
    ● Created zero-configuration Docker solution that mimics production CDC behavior locally, allowing isolated testing of database change streams without AWS costs or shared environment conflicts
    Hand Gesture Mouse Control | C++, OpenCV, MediaPipe Dec 2024 - Present
    ● Developed real-time hand tracking system using MediaPipe C++ API to detect 21 hand landmarks at 30+ FPS, implementing Kalman filtering for motion smoothing and achieving sub-100ms latency between gesture input and cursor response.
    ● Engineered custom gesture recognition algorithms mapping 3D hand coordinates to 2D screen space through perspective transformation, enabling accurate mouse control with 95% gesture detection accuracy across varying lighting conditions.
    ● Optimized performance through multi-threaded architecture separating camera capture and processing pipelines, reducing CPU usage by 40% while supporting real-time click, drag, and scroll functionalities.
    EDUCATION
    Illinois Institute of Technology, Chicago, IL Dec 2022
    Master of Computer Science
    Parul University, Vadodara, Gujarat, India May 2019
    Bachelor of Technology in Computer Science
    """

    # Initialize adaptive chunker
    chunker = AdaptiveRAGChunker(
        min_topic_size=2,  # Minimum 2 sentences per topic
        colbert_weight=0.4,
        use_global_registry=True
    )

    # Process document
    result = chunker.chunk_document(document)

    # Display results
    print(f"\n✅ Created {len(result['chunks'])} chunks")
    print(f"🔍 Discovered {result['topics_discovered']} topics automatically!")
    print(f"📊 Average coherence: {result['analysis']['avg_coherence']:.3f}")
    print(f"🎯 Average topic purity: {result['analysis'].get('avg_topic_purity', 0):.3f}")

    print("\n📚 Discovered Topics:")
    for topic_id, info in result['topic_info'].items():
        print(f"\n   Topic {topic_id}: {info['label']}")
        print(f"   Keywords: {', '.join(info['keywords'][:5])}")

    print("\n📦 Chunks:")
    for i, chunk in enumerate(result['chunks']):
        print(f"\n   Chunk {i+1} - {chunk.topic_label}")
        print(f"   Sentences: {len(chunk.sentences)}")
        print(f"   Coherence: {chunk.coherence_score:.3f}")
        print(f"   Topic confidence: {chunk.topic_probability:.3f}")
        print(f"   Keywords: {', '.join(chunk.topic_keywords[:5])}")
        print(f"   Preview: {chunk.text[:150]}...")

    # Visualize results
    print("\n\n📈 Generating visualizations...")
    chunker.visualize_results(result)

    # Test with a completely different document to show adaptability
    print("\n\n🔄 Testing with a different domain (Finance)...")

    finance_doc = """
    The stock market experienced significant volatility this quarter.
    Interest rates have been raised by the Federal Reserve.
    Cryptocurrency adoption continues to grow among institutional investors.
    Bond yields are inversely related to bond prices.
    The housing market shows signs of cooling down.

    Digital payment systems are revolutionizing financial transactions.
    Inflation remains above the central bank's target rate.
    Bitcoin and Ethereum dominate the cryptocurrency market.
    Mortgage rates have reached their highest level in years.
    Portfolio diversification is crucial for risk management.
    """

    result2 = chunker.chunk_document(finance_doc)

    print(f"\n✅ Finance document: {len(result2['chunks'])} chunks, {result2['topics_discovered']} topics")
    print("📚 Discovered Finance Topics:")
    for topic_id, info in result2['topic_info'].items():
        print(f"   Topic {topic_id}: {', '.join(info['keywords'][:3])}")

    # Show global registry
    if chunker.use_global_registry:
        print(f"\n🌍 Global Topic Registry now contains {len(chunker.topic_registry.topics)} topics")

    return result


if __name__ == "__main__":
    result = demonstrate_adaptive_chunking()
