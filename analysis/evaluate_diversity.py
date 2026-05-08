import json
import os
import argparse
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import re
from datasets import load_dataset
from huggingface_hub import login
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 可选：使用 sentence-transformers 进行语义分析
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Semantic analysis will be skipped.")


class DiversityAnalyzer:
    """分析生成问题的多样性和新颖性"""
    
    def __init__(self, original_problems: List[str], generated_problems: List[str], 
                 generated_metadata: List[Dict] = None):
        self.original_problems = original_problems
        self.generated_problems = generated_problems
        self.generated_metadata = generated_metadata or [{} for _ in generated_problems]
        
        # 合并所有问题用于某些分析
        self.all_problems = original_problems + generated_problems
        self.labels = ['original'] * len(original_problems) + ['generated'] * len(generated_problems)
        
        # 初始化结果字典
        self.results = {}
    
    # ==================== 1. N-gram 重叠分析 ====================
    
    def analyze_ngram_overlap(self, n_values: List[int] = [1, 2, 3, 4]) -> Dict:
        """分析 n-gram 重叠度"""
        print("\n" + "="*60)
        print("1. N-gram Overlap Analysis")
        print("="*60)
        
        results = {}
        
        for n in n_values:
            # 提取 n-grams
            original_ngrams = self._extract_ngrams_from_corpus(self.original_problems, n)
            generated_ngrams = self._extract_ngrams_from_corpus(self.generated_problems, n)
            
            # 计算重叠
            original_set = set(original_ngrams.keys())
            generated_set = set(generated_ngrams.keys())
            
            overlap = original_set & generated_set
            only_original = original_set - generated_set
            only_generated = generated_set - original_set
            
            # Jaccard 相似度
            jaccard = len(overlap) / len(original_set | generated_set) if (original_set | generated_set) else 0
            
            # 新颖性比例 (生成的独有 n-grams / 生成的总 n-grams)
            novelty_ratio = len(only_generated) / len(generated_set) if generated_set else 0
            
            results[f'{n}-gram'] = {
                'original_unique': len(original_set),
                'generated_unique': len(generated_set),
                'overlap_count': len(overlap),
                'only_original': len(only_original),
                'only_generated': len(only_generated),
                'jaccard_similarity': jaccard,
                'novelty_ratio': novelty_ratio,
                'top_new_ngrams': self._get_top_items(
                    {k: generated_ngrams[k] for k in only_generated}, 
                    top_k=10
                )
            }
            
            print(f"\n{n}-gram Analysis:")
            print(f"  Original unique {n}-grams: {len(original_set)}")
            print(f"  Generated unique {n}-grams: {len(generated_set)}")
            print(f"  Overlap: {len(overlap)}")
            print(f"  Jaccard Similarity: {jaccard:.4f}")
            print(f"  Novelty Ratio: {novelty_ratio:.4f}")
        
        self.results['ngram_overlap'] = results
        return results
    
    def _extract_ngrams_from_corpus(self, texts: List[str], n: int) -> Counter:
        """从语料库中提取 n-grams"""
        all_ngrams = Counter()
        for text in texts:
            words = self._tokenize(text)
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            all_ngrams.update(ngrams)
        return all_ngrams
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 转小写，移除标点，分词
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def _get_top_items(self, counter: Dict, top_k: int = 10) -> List[Tuple]:
        """获取频率最高的项"""
        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]
    
    # ==================== 2. 结构模板分析 ====================
    
    def analyze_structural_templates(self) -> Dict:
        """分析问题的结构模板"""
        print("\n" + "="*60)
        print("2. Structural Template Analysis")
        print("="*60)
        
        # 定义常见的数学问题模板
        templates = {
            'find_value': r'find\s+(the\s+)?(value|number)',
            'solve_equation': r'solve\s+(the\s+)?(equation|for)',
            'calculate': r'(calculate|compute|determine)',
            'prove': r'(prove|show\s+that)',
            'how_many': r'how\s+many',
            'what_is': r'what\s+is',
            'if_then': r'if\s+.+\s*,?\s*(then|what|find)',
            'given_that': r'given\s+(that)?',
            'let_be': r'let\s+\w+\s+(be|=)',
            'suppose': r'suppose',
            'evaluate': r'evaluate',
            'simplify': r'simplify',
            'factor': r'factor',
            'graph': r'graph',
            'area_perimeter': r'(area|perimeter|volume)',
            'probability': r'probability',
            'fraction': r'(fraction|ratio)',
            'percent': r'percent|%',
            'sequence': r'sequence|series|term',
            'function': r'function|f\(x\)',
            'inequality': r'inequality|<|>|≤|≥',
        }
        
        original_templates = self._count_templates(self.original_problems, templates)
        generated_templates = self._count_templates(self.generated_problems, templates)
        
        # 计算模板分布的 KL 散度
        template_names = list(templates.keys())
        orig_dist = np.array([original_templates.get(t, 0) for t in template_names])
        gen_dist = np.array([generated_templates.get(t, 0) for t in template_names])
        
        # 归一化
        orig_dist = orig_dist / (orig_dist.sum() + 1e-10)
        gen_dist = gen_dist / (gen_dist.sum() + 1e-10)
        
        # KL 散度 (添加平滑)
        kl_divergence = self._kl_divergence(orig_dist, gen_dist)
        
        # 新模板检测（在生成中出现但原始中很少的模板）
        new_template_patterns = self._detect_new_patterns(self.generated_problems, templates)
        
        results = {
            'original_template_counts': dict(original_templates),
            'generated_template_counts': dict(generated_templates),
            'kl_divergence': kl_divergence,
            'template_distribution_original': dict(zip(template_names, orig_dist.tolist())),
            'template_distribution_generated': dict(zip(template_names, gen_dist.tolist())),
            'new_patterns_detected': new_template_patterns[:20]
        }
        
        print(f"\nTemplate Distribution Comparison:")
        print(f"{'Template':<20} {'Original':>10} {'Generated':>10} {'Diff':>10}")
        print("-" * 50)
        for t in template_names:
            orig_count = original_templates.get(t, 0)
            gen_count = generated_templates.get(t, 0)
            diff = gen_count - orig_count
            print(f"{t:<20} {orig_count:>10} {gen_count:>10} {diff:>+10}")
        
        print(f"\nKL Divergence (template distribution): {kl_divergence:.4f}")
        
        self.results['structural_templates'] = results
        return results
    
    def _count_templates(self, problems: List[str], templates: Dict) -> Counter:
        """统计每个模板的出现次数"""
        counts = Counter()
        for problem in problems:
            problem_lower = problem.lower()
            for template_name, pattern in templates.items():
                if re.search(pattern, problem_lower):
                    counts[template_name] += 1
        return counts
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """计算 KL 散度"""
        p = p + epsilon
        q = q + epsilon
        p = p / p.sum()
        q = q / q.sum()
        return np.sum(p * np.log(p / q))
    
    def _detect_new_patterns(self, problems: List[str], existing_templates: Dict) -> List[str]:
        """检测新的结构模式"""
        # 提取不匹配现有模板的问题开头
        new_patterns = Counter()
        for problem in problems:
            problem_lower = problem.lower()
            matched = any(re.search(p, problem_lower) for p in existing_templates.values())
            if not matched:
                # 提取前 5 个词作为模式
                words = problem_lower.split()[:5]
                pattern = ' '.join(words)
                new_patterns[pattern] += 1
        
        return [p for p, _ in new_patterns.most_common(20)]
    
    # ==================== 3. 聚类分析 ====================
    
    def analyze_clusters(self, n_clusters: int = 10, method: str = 'tfidf') -> Dict:
        """聚类分析"""
        print("\n" + "="*60)
        print("3. Cluster Analysis")
        print("="*60)
        
        # 向量化
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            X = vectorizer.fit_transform(self.all_problems)
        else:
            vectorizer = CountVectorizer(max_features=5000, stop_words='english')
            X = vectorizer.fit_transform(self.all_problems)
        
        # K-Means 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # 分析每个聚类中原始和生成的比例
        n_original = len(self.original_problems)
        cluster_composition = defaultdict(lambda: {'original': 0, 'generated': 0})
        
        for i, label in enumerate(cluster_labels):
            if i < n_original:
                cluster_composition[label]['original'] += 1
            else:
                cluster_composition[label]['generated'] += 1
        
        # 计算聚类熵（多样性指标）
        original_clusters = cluster_labels[:n_original]
        generated_clusters = cluster_labels[n_original:]
        
        original_entropy = self._calculate_entropy(original_clusters)
        generated_entropy = self._calculate_entropy(generated_clusters)
        
        # 计算生成问题覆盖了多少个聚类
        original_cluster_set = set(original_clusters)
        generated_cluster_set = set(generated_clusters)
        
        cluster_coverage = len(generated_cluster_set) / n_clusters
        new_clusters = generated_cluster_set - original_cluster_set
        
        results = {
            'n_clusters': n_clusters,
            'method': method,
            'cluster_composition': dict(cluster_composition),
            'original_entropy': original_entropy,
            'generated_entropy': generated_entropy,
            'cluster_coverage': cluster_coverage,
            'clusters_only_in_generated': list(new_clusters),
            'clusters_only_in_original': list(original_cluster_set - generated_cluster_set)
        }
        
        print(f"\nCluster Composition:")
        print(f"{'Cluster':<10} {'Original':>10} {'Generated':>10} {'Ratio':>10}")
        print("-" * 40)
        for cluster_id in sorted(cluster_composition.keys()):
            orig = cluster_composition[cluster_id]['original']
            gen = cluster_composition[cluster_id]['generated']
            ratio = gen / (orig + 1e-10)
            print(f"{cluster_id:<10} {orig:>10} {gen:>10} {ratio:>10.2f}")
        
        print(f"\nEntropy (higher = more diverse):")
        print(f"  Original: {original_entropy:.4f}")
        print(f"  Generated: {generated_entropy:.4f}")
        print(f"\nCluster Coverage: {cluster_coverage:.2%}")
        
        self.results['cluster_analysis'] = results
        self.cluster_labels = cluster_labels
        self.X_vectorized = X
        
        return results
    
    def _calculate_entropy(self, labels: np.ndarray) -> float:
        """计算分布的熵"""
        counts = Counter(labels)
        total = len(labels)
        probs = np.array([count / total for count in counts.values()])
        return -np.sum(probs * np.log(probs + 1e-10))
    
    # ==================== 4. 语义相似度分析 ====================
    
    def analyze_semantic_similarity(self, sample_size: int = 500) -> Dict:
        """使用 sentence-transformers 进行语义相似度分析"""
        print("\n" + "="*60)
        print("4. Semantic Similarity Analysis")
        print("="*60)
        
        if not HAS_SENTENCE_TRANSFORMERS:
            print("Skipping semantic analysis (sentence-transformers not installed)")
            return {}
        
        # 采样以加快计算
        if len(self.original_problems) > sample_size:
            original_sample = np.random.choice(self.original_problems, sample_size, replace=False)
        else:
            original_sample = self.original_problems
        
        if len(self.generated_problems) > sample_size:
            generated_sample = np.random.choice(self.generated_problems, sample_size, replace=False)
        else:
            generated_sample = self.generated_problems
        
        print(f"Using {len(original_sample)} original and {len(generated_sample)} generated samples")
        
        # 加载模型
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 编码
        print("Encoding problems...")
        original_embeddings = model.encode(list(original_sample), show_progress_bar=True)
        generated_embeddings = model.encode(list(generated_sample), show_progress_bar=True)
        
        # 计算相似度矩阵
        print("Computing similarity matrices...")
        
        # 1. 生成与原始之间的相似度
        cross_similarity = cosine_similarity(generated_embeddings, original_embeddings)
        
        # 2. 生成内部的相似度
        gen_internal_similarity = cosine_similarity(generated_embeddings, generated_embeddings)
        np.fill_diagonal(gen_internal_similarity, 0)  # 排除自身
        
        # 3. 原始内部的相似度
        orig_internal_similarity = cosine_similarity(original_embeddings, original_embeddings)
        np.fill_diagonal(orig_internal_similarity, 0)
        
        # 统计
        results = {
            'cross_similarity': {
                'mean': float(np.mean(cross_similarity)),
                'std': float(np.std(cross_similarity)),
                'max': float(np.max(cross_similarity)),
                'min': float(np.min(cross_similarity)),
                'median': float(np.median(cross_similarity))
            },
            'generated_internal_similarity': {
                'mean': float(np.mean(gen_internal_similarity)),
                'std': float(np.std(gen_internal_similarity)),
            },
            'original_internal_similarity': {
                'mean': float(np.mean(orig_internal_similarity)),
                'std': float(np.std(orig_internal_similarity)),
            },
            # 新颖性指标：生成问题与最相似原始问题的平均距离
            'avg_min_distance_to_original': float(1 - np.mean(np.max(cross_similarity, axis=1))),
            # 多样性指标：生成问题之间的平均距离
            'avg_internal_diversity': float(1 - np.mean(gen_internal_similarity))
        }
        
        print(f"\nCross Similarity (Generated vs Original):")
        print(f"  Mean: {results['cross_similarity']['mean']:.4f}")
        print(f"  Std: {results['cross_similarity']['std']:.4f}")
        print(f"  Max: {results['cross_similarity']['max']:.4f}")
        
        print(f"\nInternal Diversity:")
        print(f"  Generated: {results['avg_internal_diversity']:.4f}")
        print(f"  Original: {1 - results['original_internal_similarity']['mean']:.4f}")
        
        print(f"\nNovelty (avg distance to nearest original):")
        print(f"  {results['avg_min_distance_to_original']:.4f}")
        
        self.results['semantic_similarity'] = results
        self.embeddings = {
            'original': original_embeddings,
            'generated': generated_embeddings
        }
        
        return results
    
    # ==================== 5. 按种子分析 ====================
    
    def analyze_by_seed(self) -> Dict:
        """按生成种子分析多样性"""
        print("\n" + "="*60)
        print("5. Analysis by Seed")
        print("="*60)
        
        if not self.generated_metadata or 'seed' not in self.generated_metadata[0]:
            print("No seed information available")
            return {}
        
        # 按种子分组
        seed_groups = defaultdict(list)
        for i, meta in enumerate(self.generated_metadata):
            seed = meta.get('seed', 'unknown')
            seed_groups[seed].append(self.generated_problems[i])
        
        results = {}
        
        for seed in sorted(seed_groups.keys()):
            problems = seed_groups[seed]
            
            # 计算内部 n-gram 多样性
            ngrams_2 = self._extract_ngrams_from_corpus(problems, 2)
            ngrams_3 = self._extract_ngrams_from_corpus(problems, 3)
            
            # 独特 n-gram 比率
            unique_2gram_ratio = len(ngrams_2) / (sum(ngrams_2.values()) + 1e-10)
            unique_3gram_ratio = len(ngrams_3) / (sum(ngrams_3.values()) + 1e-10)
            
            # 平均问题长度
            avg_length = np.mean([len(p.split()) for p in problems])
            
            results[f'seed_{seed}'] = {
                'count': len(problems),
                'unique_2gram_ratio': unique_2gram_ratio,
                'unique_3gram_ratio': unique_3gram_ratio,
                'avg_problem_length': avg_length,
                'total_unique_2grams': len(ngrams_2),
                'total_unique_3grams': len(ngrams_3)
            }
        
        print(f"\n{'Seed':<10} {'Count':>8} {'Unique 2-gram':>15} {'Unique 3-gram':>15} {'Avg Length':>12}")
        print("-" * 60)
        for seed in sorted(seed_groups.keys()):
            r = results[f'seed_{seed}']
            print(f"{seed:<10} {r['count']:>8} {r['unique_2gram_ratio']:>15.4f} "
                  f"{r['unique_3gram_ratio']:>15.4f} {r['avg_problem_length']:>12.1f}")
        
        self.results['by_seed'] = results
        return results
    
    # ==================== 6. 可视化 ====================
    
    def visualize(self, output_dir: str = 'diversity_analysis'):
        """生成可视化图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("6. Generating Visualizations")
        print("="*60)
        
        # 1. N-gram 重叠可视化
        if 'ngram_overlap' in self.results:
            self._plot_ngram_overlap(output_dir)
        
        # 2. 模板分布可视化
        if 'structural_templates' in self.results:
            self._plot_template_distribution(output_dir)
        
        # 3. 聚类可视化
        if hasattr(self, 'cluster_labels'):
            self._plot_clusters(output_dir)
        
        # 4. 语义空间可视化
        if hasattr(self, 'embeddings'):
            self._plot_semantic_space(output_dir)
        
        print(f"\nVisualizations saved to {output_dir}/")
    
    def _plot_ngram_overlap(self, output_dir: str):
        """绘制 n-gram 重叠图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ngram_data = self.results['ngram_overlap']
        n_values = [k.replace('-gram', '') for k in ngram_data.keys()]
        
        # Jaccard 相似度
        jaccard_scores = [ngram_data[f'{n}-gram']['jaccard_similarity'] for n in n_values]
        axes[0].bar(n_values, jaccard_scores, color='steelblue')
        axes[0].set_xlabel('N-gram')
        axes[0].set_ylabel('Jaccard Similarity')
        axes[0].set_title('N-gram Overlap (Jaccard Similarity)')
        axes[0].set_ylim(0, 1)
        
        # 新颖性比例
        novelty_ratios = [ngram_data[f'{n}-gram']['novelty_ratio'] for n in n_values]
        axes[1].bar(n_values, novelty_ratios, color='coral')
        axes[1].set_xlabel('N-gram')
        axes[1].set_ylabel('Novelty Ratio')
        axes[1].set_title('Generated Novelty Ratio')
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ngram_overlap.png', dpi=150)
        plt.close()
        print(f"  Saved: {output_dir}/ngram_overlap.png")
    
    def _plot_template_distribution(self, output_dir: str):
        """绘制模板分布对比图"""
        template_data = self.results['structural_templates']
        
        templates = list(template_data['original_template_counts'].keys())
        orig_counts = [template_data['original_template_counts'].get(t, 0) for t in templates]
        gen_counts = [template_data['generated_template_counts'].get(t, 0) for t in templates]
        
        # 归一化
        orig_norm = np.array(orig_counts) / (sum(orig_counts) + 1e-10)
        gen_norm = np.array(gen_counts) / (sum(gen_counts) + 1e-10)
        
        x = np.arange(len(templates))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width/2, orig_norm, width, label='Original', color='steelblue')
        ax.bar(x + width/2, gen_norm, width, label='Generated', color='coral')
        
        ax.set_ylabel('Proportion')
        ax.set_title('Structural Template Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(templates, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/template_distribution.png', dpi=150)
        plt.close()
        print(f"  Saved: {output_dir}/template_distribution.png")
    
    def _plot_clusters(self, output_dir: str):
        """绘制聚类分布图"""
        cluster_data = self.results['cluster_analysis']
        
        # 降维可视化
        X_dense = self.X_vectorized.toarray() if hasattr(self.X_vectorized, 'toarray') else self.X_vectorized
        
        # 使用 PCA 降维
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_dense)
        
        n_original = len(self.original_problems)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 按来源着色
        colors = ['steelblue' if i < n_original else 'coral' for i in range(len(X_2d))]
        axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.5, s=10)
        axes[0].set_title('PCA Visualization (Blue=Original, Red=Generated)')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        
        # 按聚类着色
        scatter = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=self.cluster_labels, 
                                  cmap='tab10', alpha=0.5, s=10)
        axes[1].set_title('Cluster Visualization')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cluster_visualization.png', dpi=150)
        plt.close()
        print(f"  Saved: {output_dir}/cluster_visualization.png")
    
    def _plot_semantic_space(self, output_dir: str):
        """绘制语义空间图"""
        if not hasattr(self, 'embeddings'):
            return
        
        # 合并 embeddings
        all_embeddings = np.vstack([self.embeddings['original'], self.embeddings['generated']])
        labels = ['Original'] * len(self.embeddings['original']) + \
                 ['Generated'] * len(self.embeddings['generated'])
        
        # t-SNE 降维
        print("  Running t-SNE (this may take a while)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(all_embeddings)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        n_orig = len(self.embeddings['original'])
        ax.scatter(X_tsne[:n_orig, 0], X_tsne[:n_orig, 1], 
                  c='steelblue', label='Original', alpha=0.5, s=20)
        ax.scatter(X_tsne[n_orig:, 0], X_tsne[n_orig:, 1], 
                  c='coral', label='Generated', alpha=0.5, s=20)
        
        ax.set_title('Semantic Space Visualization (t-SNE)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/semantic_space.png', dpi=150)
        plt.close()
        print(f"  Saved: {output_dir}/semantic_space.png")
    
    # ==================== 7. 保存结果 ====================
    
    def _convert_to_serializable(self, obj):
        """递归转换对象为可 JSON 序列化的格式"""
        if isinstance(obj, dict):
            # 转换字典的键和值
            return {
                str(k) if isinstance(k, (np.integer, np.floating)) else k: 
                self._convert_to_serializable(v) 
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def save_results(self, output_path: str):
        """保存分析结果"""
        # 转换为可序列化格式
        serializable_results = self._convert_to_serializable(self.results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_path}")
    
    # ==================== 8. 完整分析 ====================
    
    def run_full_analysis(self, output_dir: str = 'diversity_analysis'):
        """运行完整分析"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("DIVERSITY AND NOVELTY ANALYSIS")
        print("="*70)
        print(f"Original problems: {len(self.original_problems)}")
        print(f"Generated problems: {len(self.generated_problems)}")
        
        # 1. N-gram 分析
        self.analyze_ngram_overlap()
        
        # 2. 结构模板分析
        self.analyze_structural_templates()
        
        # 3. 聚类分析
        self.analyze_clusters(n_clusters=10)
        
        # 4. 语义相似度分析
        self.analyze_semantic_similarity(sample_size=500)
        
        # 5. 按种子分析
        self.analyze_by_seed()
        
        # 6. 可视化
        self.visualize(output_dir)
        
        # 7. 保存结果
        self.save_results(f'{output_dir}/analysis_results.json')
        
        # 8. 打印摘要
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """打印分析摘要"""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        print("\n📊 Diversity Metrics:")
        
        if 'ngram_overlap' in self.results:
            ngram = self.results['ngram_overlap']
            print(f"  - 2-gram Novelty Ratio: {ngram['2-gram']['novelty_ratio']:.4f}")
            print(f"  - 3-gram Novelty Ratio: {ngram['3-gram']['novelty_ratio']:.4f}")
        
        if 'cluster_analysis' in self.results:
            cluster = self.results['cluster_analysis']
            print(f"  - Cluster Coverage: {cluster['cluster_coverage']:.2%}")
            print(f"  - Generated Entropy: {cluster['generated_entropy']:.4f}")
        
        if 'semantic_similarity' in self.results:
            semantic = self.results['semantic_similarity']
            print(f"  - Semantic Novelty: {semantic['avg_min_distance_to_original']:.4f}")
            print(f"  - Internal Diversity: {semantic['avg_internal_diversity']:.4f}")
        
        print("\n🎯 Interpretation:")
        print("  - Higher Novelty Ratio → More unique n-grams in generated problems")
        print("  - Higher Cluster Coverage → Generated problems cover more topic areas")
        print("  - Higher Semantic Novelty → Generated problems are more different from originals")
        print("  - Higher Internal Diversity → Less repetition within generated problems")


def main(args):
    """主函数"""
    STORAGE_PATH = os.getenv("STORAGE_PATH")
    
    # 1. 加载原始 math12k 数据集
    print("Loading original math12k dataset...")
    from datasets import load_dataset
    original_dataset = load_dataset("hiyouga/math12k", split="train")
    original_questions = [item["problem"] for item in original_dataset]
    original_answers = [item["answer"] for item in original_dataset]
    
    # 2. 重现种子采样逻辑，收集所有被选中的原始题目
    import random
    original_problems = []
    original_metadata = []
    
    for seed in range(8):
        title_random = random.Random(seed)
        if args.num_samples_per_seed > len(original_questions):
            selected_indices = title_random.choices(range(len(original_questions)), k=args.num_samples_per_seed)
        else:
            selected_indices = title_random.sample(range(len(original_questions)), args.num_samples_per_seed)
        
        for idx in selected_indices:
            original_problems.append(original_questions[idx])
            original_metadata.append({
                'source': 'original',
                'answer': original_answers[idx],
                'seed': seed
            })
    
    print(f"Collected {len(original_problems)} original problems from seeds 0-7")
    
    # 3. 加载生成的题目
    generated_problems = []
    generated_metadata = []
    
    # 从传入的文件路径解析 experiment_name
    # 例如: $STORAGE_PATH/generated_question/Qwen3-4B-Base_solver_..._0.json
    # 提取 experiment_name = Qwen3-4B-Base_solver_...
    
    if args.generated_file:
        # 直接指定了文件路径，解析出 experiment_name 和 base_path
        import os.path as osp
        base_dir = osp.dirname(args.generated_file)
        filename = osp.basename(args.generated_file)
        
        # 从文件名解析 experiment_name (去掉 _0.json 等后缀)
        # 格式: {experiment_name}_{seed}_results.json 或 {experiment_name}_{seed}.json
        parts = filename.rsplit('_', 1)
        if len(parts) == 2:
            experiment_name = parts[0]
            # 检查是否是 _results.json 格式
            if experiment_name.endswith('_results'):
                experiment_name = experiment_name[:-8]  # 去掉 _results
        else:
            experiment_name = filename.replace('.json', '')
        
        print(f"Detected experiment_name: {experiment_name}")
        print(f"Base directory: {base_dir}")
        
        # 加载所有 seed 的文件
        for seed in range(8):
            # 尝试不同的文件名格式
            possible_files = [
                f'{base_dir}/{experiment_name}_{seed}_results.json',
                f'{base_dir}/{experiment_name}_{seed}.json',
            ]
            
            file_found = False
            for file_path in possible_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        for item in data:
                            # 过滤条件
                            score = item.get('score', 0)
                            answer = item.get('answer', '')
                            question = item.get('question', '')
                            
                            if (score >= args.min_score and 
                                score <= args.max_score and 
                                answer and answer != 'None' and
                                question and question.strip()):
                                
                                generated_problems.append(question)
                                generated_metadata.append({
                                    'source': 'generated',
                                    'answer': answer,
                                    'seed': seed,
                                    'score': score
                                })
                        
                        print(f"  Loaded seed {seed}: {len([m for m in generated_metadata if m.get('seed') == seed])} problems")
                        file_found = True
                        break
                except FileNotFoundError:
                    continue
                except json.JSONDecodeError as e:
                    print(f"  Warning: Failed to parse {file_path}: {e}")
                    continue
            
            if not file_found:
                print(f"  Warning: No file found for seed {seed}")
    else:
        # 使用 experiment_name 参数
        experiment_name = args.experiment_name
        base_dir = f'{STORAGE_PATH}/generated_question'
        
        for seed in range(8):
            file_path = f'{base_dir}/{experiment_name}_{seed}_results.json'
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        score = item.get('score', 0)
                        answer = item.get('answer', '')
                        question = item.get('question', '')
                        
                        if (score >= args.min_score and 
                            score <= args.max_score and 
                            answer and answer != 'None' and
                            question and question.strip()):
                            
                            generated_problems.append(question)
                            generated_metadata.append({
                                'source': 'generated',
                                'answer': answer,
                                'seed': seed,
                                'score': score
                            })
                    
                    print(f"  Loaded seed {seed}: {len([m for m in generated_metadata if m.get('seed') == seed])} problems")
            except FileNotFoundError:
                print(f"  Warning: File {experiment_name}_{seed}_results.json not found")
                continue
    
    print(f"\nCollected {len(generated_problems)} generated problems")
    
    print(f"\nDataset Summary:")
    print(f"  Original problems: {len(original_problems)}")
    print(f"  Generated problems: {len(generated_problems)}")
    print(f"  Total: {len(original_problems) + len(generated_problems)}")
    
    if len(generated_problems) == 0:
        print("Error: No generated problems found!")
        return
    
    # 4. 运行分析
    analyzer = DiversityAnalyzer(
        original_problems=original_problems,
        generated_problems=generated_problems,
        generated_metadata=generated_metadata
    )
    
    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    elif args.generated_file:
        import os.path as osp
        base_dir = osp.dirname(args.generated_file)
        filename = osp.basename(args.generated_file)
        parts = filename.rsplit('_', 1)
        if len(parts) == 2:
            exp_name = parts[0]
            if exp_name.endswith('_results'):
                exp_name = exp_name[:-8]
        else:
            exp_name = filename.replace('.json', '')
        output_dir = f'{base_dir}/diversity_analysis_{exp_name}'
    else:
        output_dir = f'{STORAGE_PATH}/diversity_analysis/{args.experiment_name}'
    
    results = analyzer.run_full_analysis(output_dir=output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze diversity and novelty of generated problems")
    
    parser.add_argument("--generated_file", type=str, default=None,
                       help="Path to a generated file (e.g., /path/to/experiment_0.json). Will auto-detect experiment name and load all seeds.")
    
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Experiment name (used when --generated_file is not provided)")
    
    parser.add_argument("--num_samples_per_seed", type=int, default=1000,
                       help="Number of original samples per seed")
    
    parser.add_argument("--min_score", type=float, default=0.0,
                       help="Minimum score for generated problems")
    
    parser.add_argument("--max_score", type=float, default=1.0,
                       help="Maximum score for generated problems")
    
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    main(args)