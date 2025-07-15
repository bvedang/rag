"""
Adaptive Code Analyzer: BERTopic + AST for Automatic Test Generation
===================================================================
Extends the adaptive chunking concept to analyze code patterns and generate unit tests
"""

import ast
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import astunparse

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class CodePattern:
    """Represents a discovered code pattern"""
    pattern_id: int
    pattern_type: str  # e.g., "error_handling", "data_validation", "api_call"
    keywords: List[str]
    example_snippets: List[str]
    test_strategies: List[str] = field(default_factory=list)
    complexity_score: float = 0.0

@dataclass
class CodeChunk:
    """Enhanced chunk for code analysis"""
    code: str
    ast_node: Optional[ast.AST] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    calls_made: List[str] = field(default_factory=list)
    pattern_id: Optional[int] = None
    pattern_type: Optional[str] = None
    pattern_keywords: List[str] = field(default_factory=list)
    complexity: int = 0
    test_cases: List[str] = field(default_factory=list)

class CodePatternAnalyzer:
    """Analyzes code to discover patterns and generate tests"""

    def __init__(self, embedding_model: str = "all-mpnet-base-v2"):
        self.sentence_model = SentenceTransformer(embedding_model)
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            min_topic_size=2,
            n_gram_range=(1, 3),
            calculate_probabilities=True
        )

        # Pattern templates for test generation
        self.test_templates = {
            "error_handling": [
                "Test exception handling",
                "Test error messages",
                "Test recovery behavior"
            ],
            "data_validation": [
                "Test with valid input",
                "Test with invalid input",
                "Test edge cases",
                "Test type checking"
            ],
            "api_call": [
                "Mock external calls",
                "Test timeout handling",
                "Test response parsing",
                "Test authentication"
            ],
            "database": [
                "Test CRUD operations",
                "Test transaction handling",
                "Test connection errors"
            ],
            "async": [
                "Test async execution",
                "Test concurrent operations",
                "Test cancellation"
            ],
            "algorithm": [
                "Test with empty input",
                "Test with single element",
                "Test with large dataset",
                "Test performance"
            ]
        }

    def extract_code_features(self, node: ast.AST) -> Dict[str, Any]:
        """Extract features from AST node"""
        features = {
            "has_try_except": False,
            "has_loops": False,
            "has_conditionals": False,
            "has_return": False,
            "has_yield": False,
            "has_async": False,
            "has_decorators": False,
            "calls": [],
            "imports": [],
            "raises": [],
            "complexity": 0
        }

        class FeatureVisitor(ast.NodeVisitor):
            def __init__(self, features):
                self.features = features
                self.complexity = 0

            def visit_Try(self, node):
                self.features["has_try_except"] = True
                self.complexity += 2
                self.generic_visit(node)

            def visit_For(self, node):
                self.features["has_loops"] = True
                self.complexity += 1
                self.generic_visit(node)

            def visit_While(self, node):
                self.features["has_loops"] = True
                self.complexity += 1
                self.generic_visit(node)

            def visit_If(self, node):
                self.features["has_conditionals"] = True
                self.complexity += 1
                self.generic_visit(node)

            def visit_Return(self, node):
                self.features["has_return"] = True
                self.generic_visit(node)

            def visit_Yield(self, node):
                self.features["has_yield"] = True
                self.complexity += 1
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                self.features["has_async"] = True
                self.complexity += 1
                self.generic_visit(node)

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    self.features["calls"].append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    self.features["calls"].append(node.func.attr)
                self.generic_visit(node)

            def visit_Raise(self, node):
                if node.exc and isinstance(node.exc, ast.Call):
                    if isinstance(node.exc.func, ast.Name):
                        self.features["raises"].append(node.exc.func.id)
                self.generic_visit(node)

        visitor = FeatureVisitor(features)
        visitor.visit(node)
        features["complexity"] = visitor.complexity

        return features

    def code_to_text_representation(self, chunk: CodeChunk) -> str:
        """Convert code chunk to text for embedding"""
        parts = []

        # Add function/class name
        if chunk.function_name:
            parts.append(f"function {chunk.function_name}")
        if chunk.class_name:
            parts.append(f"class {chunk.class_name}")

        # Add code structure features
        features = self.extract_code_features(chunk.ast_node) if chunk.ast_node else {}

        if features.get("has_try_except"):
            parts.append("error handling exception")
        if features.get("has_loops"):
            parts.append("iteration loop")
        if features.get("has_conditionals"):
            parts.append("conditional logic")
        if features.get("has_async"):
            parts.append("asynchronous async await")
        if features.get("has_yield"):
            parts.append("generator yield")

        # Add calls made
        for call in features.get("calls", []):
            parts.append(f"calls {call}")

        # Add raises
        for exc in features.get("raises", []):
            parts.append(f"raises {exc}")

        # Add docstring if available
        if chunk.ast_node and ast.get_docstring(chunk.ast_node):
            doc = ast.get_docstring(chunk.ast_node)
            parts.append(doc[:100])  # First 100 chars of docstring

        return " ".join(parts)

    def identify_pattern_type(self, chunk: CodeChunk, features: Dict) -> str:
        """Identify the primary pattern type of a code chunk"""
        # Simple heuristic-based classification
        if features.get("has_try_except") or features.get("raises"):
            return "error_handling"
        elif any(call in ["isinstance", "type", "validate", "check"] for call in features.get("calls", [])):
            return "data_validation"
        elif any(call in ["get", "post", "request", "fetch"] for call in features.get("calls", [])):
            return "api_call"
        elif any(call in ["query", "execute", "commit", "rollback"] for call in features.get("calls", [])):
            return "database"
        elif features.get("has_async"):
            return "async"
        elif features.get("has_loops") and features.get("complexity", 0) > 3:
            return "algorithm"
        else:
            return "general"

    def generate_test_cases(self, chunk: CodeChunk) -> List[str]:
        """Generate test cases based on discovered patterns"""
        test_cases = []

        # Get function signature
        if not chunk.function_name:
            return test_cases

        # Extract parameters
        params = []
        if chunk.ast_node and isinstance(chunk.ast_node, ast.FunctionDef):
            params = [arg.arg for arg in chunk.ast_node.args.args if arg.arg != 'self']

        # Generate tests based on pattern
        pattern_type = chunk.pattern_type or "general"
        test_strategies = self.test_templates.get(pattern_type, ["Test basic functionality"])

        for strategy in test_strategies:
            test_name = f"test_{chunk.function_name}_{strategy.lower().replace(' ', '_')}"

            # Generate test code
            test_code = f'''
def {test_name}():
    """Test: {strategy} for {chunk.function_name}"""
'''

            # Add pattern-specific test logic
            if pattern_type == "error_handling":
                test_code += f'''    # Test error handling
    with pytest.raises(Exception):
        {chunk.function_name}(invalid_input)

    # Test valid input
    result = {chunk.function_name}(valid_input)
    assert result is not None
'''
            elif pattern_type == "data_validation":
                test_code += f'''    # Test with valid data
    valid_data = {self._generate_sample_data(params)}
    result = {chunk.function_name}(**valid_data)
    assert result is not None

    # Test with invalid data
    invalid_data = {self._generate_invalid_data(params)}
    with pytest.raises((ValueError, TypeError)):
        {chunk.function_name}(**invalid_data)
'''
            elif pattern_type == "api_call":
                test_code += f'''    # Mock external API
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {{"status": "success"}}
        result = {chunk.function_name}()
        assert result is not None
        mock_get.assert_called_once()
'''
            elif pattern_type == "async":
                test_code = f'''
@pytest.mark.asyncio
async def {test_name}():
    """Test: {strategy} for async {chunk.function_name}"""
    result = await {chunk.function_name}()
    assert result is not None
'''
            else:
                # Generic test
                test_code += f'''    # Basic functionality test
    result = {chunk.function_name}({', '.join(['sample_' + p for p in params])})
    assert result is not None
'''

            test_cases.append(test_code.strip())

        return test_cases

    def _generate_sample_data(self, params: List[str]) -> str:
        """Generate sample valid data based on parameter names"""
        data = {}
        for param in params:
            if 'id' in param:
                data[param] = "123"
            elif 'name' in param:
                data[param] = "'test_name'"
            elif 'email' in param:
                data[param] = "'test@example.com'"
            elif 'count' in param or 'num' in param:
                data[param] = "5"
            elif 'list' in param or 'items' in param:
                data[param] = "[1, 2, 3]"
            else:
                data[param] = "'sample_value'"
        return str(data)

    def _generate_invalid_data(self, params: List[str]) -> str:
        """Generate sample invalid data"""
        data = {}
        for param in params:
            if 'id' in param:
                data[param] = "None"
            elif 'email' in param:
                data[param] = "'invalid-email'"
            elif 'count' in param or 'num' in param:
                data[param] = "-1"
            else:
                data[param] = "None"
        return str(data)

    def analyze_code_file(self, code: str) -> Dict[str, Any]:
        """Main method to analyze code and generate tests"""
        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}

        # Extract functions and classes
        chunks = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                chunk = CodeChunk(
                    code=astunparse.unparse(node),
                    ast_node=node,
                    function_name=node.name,
                    complexity=self.extract_code_features(node)["complexity"]
                )
                chunks.append(chunk)
            elif isinstance(node, ast.ClassDef):
                # Process methods within classes
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        chunk = CodeChunk(
                            code=astunparse.unparse(item),
                            ast_node=item,
                            function_name=item.name,
                            class_name=node.name,
                            complexity=self.extract_code_features(item)["complexity"]
                        )
                        chunks.append(chunk)

        if not chunks:
            return {"error": "No functions found in code"}

        # Create text representations for embedding
        texts = [self.code_to_text_representation(chunk) for chunk in chunks]

        # Discover patterns using BERTopic
        if len(texts) > 1:
            embeddings = self.sentence_model.encode(texts)
            topics, probs = self.topic_model.fit_transform(texts, embeddings)

            # Assign patterns to chunks
            for i, (chunk, topic) in enumerate(zip(chunks, topics)):
                if topic != -1:
                    chunk.pattern_id = topic
                    # Get topic keywords
                    topic_words = self.topic_model.get_topic(topic)
                    if topic_words:
                        chunk.pattern_keywords = [word for word, _ in topic_words[:5]]

                # Identify pattern type
                features = self.extract_code_features(chunk.ast_node)
                chunk.pattern_type = self.identify_pattern_type(chunk, features)

                # Generate test cases
                chunk.test_cases = self.generate_test_cases(chunk)
        else:
            # Single function - analyze directly
            chunk = chunks[0]
            features = self.extract_code_features(chunk.ast_node)
            chunk.pattern_type = self.identify_pattern_type(chunk, features)
            chunk.test_cases = self.generate_test_cases(chunk)

        # Create test suite
        test_suite = self._create_test_suite(chunks)

        # Analysis summary
        pattern_distribution = {}
        for chunk in chunks:
            pattern = chunk.pattern_type or "general"
            pattern_distribution[pattern] = pattern_distribution.get(pattern, 0) + 1

        return {
            "chunks": chunks,
            "test_suite": test_suite,
            "analysis": {
                "total_functions": len(chunks),
                "pattern_distribution": pattern_distribution,
                "avg_complexity": np.mean([c.complexity for c in chunks]),
                "test_coverage": sum(len(c.test_cases) for c in chunks)
            }
        }

    def _create_test_suite(self, chunks: List[CodeChunk]) -> str:
        """Create a complete test suite"""
        imports = set()
        test_classes = {}

        # Group by class
        for chunk in chunks:
            class_name = chunk.class_name or "General"
            if class_name not in test_classes:
                test_classes[class_name] = []
            test_classes[class_name].append(chunk)

            # Collect necessary imports
            if chunk.pattern_type == "error_handling":
                imports.add("import pytest")
            if chunk.pattern_type == "api_call":
                imports.add("from unittest.mock import patch")
            if chunk.pattern_type == "async":
                imports.add("import pytest")
                imports.add("import asyncio")

        # Build test suite
        test_suite = "# Auto-generated test suite using pattern discovery\n\n"
        test_suite += "\n".join(sorted(imports)) + "\n\n"

        for class_name, class_chunks in test_classes.items():
            if class_name != "General":
                test_suite += f"\nclass Test{class_name}:\n"
                indent = "    "
            else:
                test_suite += "\n# Standalone function tests\n"
                indent = ""

            for chunk in class_chunks:
                for test_case in chunk.test_cases:
                    # Indent test cases for classes
                    indented_test = "\n".join(
                        indent + line if line else line
                        for line in test_case.split("\n")
                    )
                    test_suite += "\n" + indented_test + "\n"

        return test_suite


def demonstrate_code_analysis():
    """Demonstrate code analysis and test generation"""

    print("🚀 Adaptive Code Analysis with Automatic Test Generation")
    print("=" * 60)

    # Sample code to analyze
    sample_code = '''
import requests
import json
from typing import List, Dict, Optional

class UserService:
    """Service for managing users"""

    def __init__(self, api_url: str):
        self.api_url = api_url
        self.session = requests.Session()

    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        if not email or '@' not in email:
            return False

        parts = email.split('@')
        if len(parts) != 2:
            return False

        return '.' in parts[1]

    def get_user(self, user_id: int) -> Optional[Dict]:
        """Fetch user from API"""
        try:
            response = self.session.get(f"{self.api_url}/users/{user_id}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching user: {e}")
            return None

    async def process_users_batch(self, user_ids: List[int]) -> List[Dict]:
        """Process multiple users asynchronously"""
        results = []
        for user_id in user_ids:
            user = self.get_user(user_id)
            if user:
                results.append(user)
        return results

def calculate_fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number"""
    if n <= 0:
        raise ValueError("n must be positive")
    elif n == 1:
        return 0
    elif n == 2:
        return 1

    a, b = 0, 1
    for _ in range(2, n):
        a, b = b, a + b

    return b

def parse_config(config_str: str) -> Dict:
    """Parse configuration string"""
    try:
        config = json.loads(config_str)

        # Validate required fields
        required_fields = ['version', 'settings']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
'''

    # Initialize analyzer
    analyzer = CodePatternAnalyzer()

    # Analyze code
    print("\n📝 Analyzing code patterns...")
    result = analyzer.analyze_code_file(sample_code)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    # Display results
    print("\n✅ Analysis complete!")
    print(f"📊 Found {result['analysis']['total_functions']} functions")
    print(f"🔍 Average complexity: {result['analysis']['avg_complexity']:.2f}")
    print(f"🧪 Generated {result['analysis']['test_coverage']} test cases")

    print("\n📈 Pattern Distribution:")
    for pattern, count in result['analysis']['pattern_distribution'].items():
        print(f"   • {pattern}: {count} functions")

    print("\n🔍 Discovered Patterns:")
    for chunk in result['chunks']:
        print(f"\n   Function: {chunk.function_name}")
        if chunk.class_name:
            print(f"   Class: {chunk.class_name}")
        print(f"   Pattern: {chunk.pattern_type}")
        print(f"   Complexity: {chunk.complexity}")
        if chunk.pattern_keywords:
            print(f"   Keywords: {', '.join(chunk.pattern_keywords)}")
        print(f"   Tests generated: {len(chunk.test_cases)}")

    # Display generated test suite
    print("\n\n🧪 Generated Test Suite:")
    print("-" * 60)
    print(result['test_suite'])

    return result


if __name__ == "__main__":
    result = demonstrate_code_analysis()
