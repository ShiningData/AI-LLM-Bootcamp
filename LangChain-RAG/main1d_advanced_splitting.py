"""
Advanced Text Splitting - Part 1d

Shows how to:
- Handle code-aware text splitting
- Split documents with mixed content types
- Optimize splitting for different content patterns
- Preserve semantic boundaries in complex documents
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document

def create_code_document():
    """Create a document with mixed code and text content."""
    content = """# Data Processing Pipeline

This section explains our data processing approach for handling large-scale datasets.

## Overview

Our pipeline consists of three main stages: ingestion, transformation, and output.

```python
def process_data(raw_data):
    \"\"\"
    Main data processing function.
    
    Args:
        raw_data: Input data in various formats
    
    Returns:
        processed_data: Cleaned and transformed data
    \"\"\"
    # Clean the data
    cleaned_data = clean_missing_values(raw_data)
    
    # Transform features
    transformed_data = apply_transformations(cleaned_data)
    
    # Validate results
    if validate_data(transformed_data):
        return transformed_data
    else:
        raise ValidationError("Data validation failed")
```

The process_data function handles the main pipeline logic and includes comprehensive error handling.

## Error Handling

We implement several layers of error handling:

```python
try:
    result = process_data(input_data)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    return None
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

## Configuration

Use the following configuration for optimal performance:

```yaml
pipeline:
  batch_size: 1000
  timeout: 300
  retry_count: 3
  validation:
    enabled: true
    strict_mode: false
```

This configuration ensures reliable processing while maintaining good performance characteristics."""
    
    return content

def create_mixed_format_document():
    """Create a document with multiple content formats."""
    content = """# TechCorp API Documentation

Welcome to the TechCorp API. This guide covers authentication, endpoints, and examples.

## Authentication

All API requests require authentication using OAuth2. Include your token in the Authorization header:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     -X GET "https://api.techcorp.com/v1/users"
```

## User Management

### Create User

**Endpoint:** `POST /v1/users`

**Request Body:**
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "role": "user",
  "preferences": {
    "notifications": true,
    "theme": "dark"
  }
}
```

**Response:**
```json
{
  "id": "user_123",
  "username": "john_doe",
  "email": "john@example.com",
  "created_at": "2024-01-01T10:00:00Z",
  "status": "active"
}
```

### Error Codes

| Code | Description | Action |
|------|-------------|---------|
| 400  | Bad Request | Check request format |
| 401  | Unauthorized | Verify authentication |
| 429  | Rate Limited | Implement backoff |
| 500  | Server Error | Retry or contact support |

## Rate Limiting

Our API implements rate limiting to ensure fair usage:

- **Basic Plan:** 1,000 requests per hour
- **Professional:** 5,000 requests per hour  
- **Enterprise:** 20,000 requests per hour

When you exceed limits, the API returns HTTP 429 with retry-after headers."""

    return content

def demonstrate_code_aware_splitting():
    """Show how to split code documents while preserving structure."""
    print("=== Code-Aware Text Splitting ===\n")
    
    code_content = create_code_document()
    code_doc = Document(
        page_content=code_content,
        metadata={"type": "technical_doc", "contains_code": True}
    )
    
    print(f"Original document: {len(code_content)} characters")
    print(f"Contains code blocks: {'```' in code_content}")
    print()
    
    # Strategy 1: Standard recursive splitting
    print("Strategy 1: Standard Recursive Splitting")
    standard_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    
    standard_chunks = standard_splitter.split_documents([code_doc])
    print(f"Created {len(standard_chunks)} chunks")
    
    for i, chunk in enumerate(standard_chunks[:3], 1):
        # Check if this chunk contains code blocks (markdown code fences)
        has_code = '```' in chunk.page_content
        print(f"  Chunk {i}: {'Code' if has_code else 'Text'}")
        print(f"    Length: {len(chunk.page_content)} chars")
        print(f"    Preview: {chunk.page_content[:60]}...")
    print()
    
    # Strategy 2: Code-aware splitting with custom separators
    print("Strategy 2: Code-Aware Splitting")
    code_aware_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=[
            "\n## ",      # Section headers
            "\n### ",     # Subsection headers  
            "\n```\n",    # End of code blocks
            "\n\n",       # Paragraphs
            "\n",         # Lines
            ". ",         # Sentences
            " ",          # Words
            ""            # Characters
        ]
    )
    
    code_chunks = code_aware_splitter.split_documents([code_doc])
    print(f"Created {len(code_chunks)} chunks with code awareness")
    
    # Analyze the first 3 chunks from the code-aware splitter to show content classification
    for i, chunk in enumerate(code_chunks[:3], 1):
        # Check if chunk contains code blocks (markdown ``` fences)
        has_code = '```' in chunk.page_content
        # Check if chunk starts with a markdown header (# ## ###)
        starts_section = chunk.page_content.strip().startswith('#')
        
        # Display chunk with appropriate label: Header for headers, Code for code, Text for plain text
        print(f"  Chunk {i}: {'Header' if starts_section else 'Code' if has_code else 'Text'}")
        print(f"    Length: {len(chunk.page_content)} chars")
        # Preview with newlines replaced by spaces for single-line display
        print(f"    Preview: {chunk.page_content[:60].replace(chr(10), ' ')}...")
        print(f"    Starts section: {starts_section}")
        print(f"    Contains code: {has_code}")
    print()  # Empty line for spacing between sections

def demonstrate_mixed_format_splitting():
    """Show splitting strategies for documents with multiple formats."""
    print("=== Mixed Format Document Splitting ===\n")
    
    mixed_content = create_mixed_format_document()
    mixed_doc = Document(
        page_content=mixed_content,
        metadata={"type": "api_documentation", "formats": ["markdown", "json", "bash", "table"]}
    )
    
    print(f"Mixed format document: {len(mixed_content)} characters")
    print("Contains: Markdown headers, JSON examples, bash commands, tables")
    print()
    
    # Strategy 1: Preserve sections
    print("Strategy 1: Section-Preserving Split")
    section_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=[
            "\n## ",      # Major sections
            "\n### ",     # Subsections
            "\n\n```",    # Code block starts
            "\n```\n",    # Code block ends
            "\n\n",       # Paragraphs
            "\n|",        # Table rows
            "\n",         # Lines
            " ",          # Words
        ]
    )
    
    section_chunks = section_splitter.split_documents([mixed_doc])
    print(f"Created {len(section_chunks)} section-aware chunks")
    
    for i, chunk in enumerate(section_chunks[:4], 1):
        has_json = '{' in chunk.page_content and '}' in chunk.page_content
        has_table = '|' in chunk.page_content
        has_code = '```' in chunk.page_content
        starts_section = chunk.page_content.strip().startswith('#')
        
        content_type = 'Header' if starts_section else 'Table' if has_table else 'Code' if has_code else 'JSON' if has_json else 'Text'
        print(f"  Chunk {i}: {content_type}")
        print(f"    Length: {len(chunk.page_content)} chars")
        first_line = chunk.page_content.split('\n')[0][:50]
        print(f"    Starts: {first_line}...")
        print(f"    Content: {'Section' if starts_section else 'Table' if has_table else 'Code' if has_code else 'JSON' if has_json else 'Text'}")
    print()

def demonstrate_semantic_boundary_preservation():
    """Show how to preserve semantic boundaries during splitting."""
    print("=== Semantic Boundary Preservation ===\n")
    
    # Create content with clear semantic boundaries
    semantic_content = """Machine Learning Fundamentals

Introduction
Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming. It relies on algorithms that can identify patterns in data and make predictions.

Supervised Learning
Supervised learning uses labeled training data to learn a mapping function. The algorithm learns from input-output pairs to predict outcomes for new data. Common types include:
- Classification: Predicting categories (email spam detection)
- Regression: Predicting continuous values (house price prediction)

Unsupervised Learning  
Unsupervised learning finds hidden patterns in data without labeled examples. The algorithm must discover structure on its own. Key approaches include:
- Clustering: Grouping similar data points (customer segmentation)
- Dimensionality reduction: Simplifying data while preserving information

Reinforcement Learning
Reinforcement learning trains agents through interaction with an environment. The agent learns by receiving rewards or penalties for actions. Applications include:
- Game playing (Chess, Go)
- Robotics (Navigation, manipulation)
- Autonomous systems (Self-driving cars)

Evaluation Metrics
Model performance is measured using various metrics depending on the problem type:
- Accuracy: Percentage of correct predictions
- Precision: True positives divided by predicted positives  
- Recall: True positives divided by actual positives
- F1-Score: Harmonic mean of precision and recall"""

    semantic_doc = Document(
        page_content=semantic_content,
        metadata={"type": "educational", "subject": "machine_learning"}
    )
    
    print("Semantic boundary preservation strategies:")
    print()
    
    # Strategy 1: Topic-based splitting
    print("Strategy 1: Topic-Based Splitting")
    topic_splitter = CharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separator="\n\n"  # Split on double newlines (topic boundaries)
    )
    
    topic_chunks = topic_splitter.split_documents([semantic_doc])
    print(f"Created {len(topic_chunks)} topic-based chunks")
    
    for i, chunk in enumerate(topic_chunks[:3], 1):
        first_line = chunk.page_content.strip().split('\n')[0]
        is_heading = not first_line[0].islower() and len(first_line.split()) <= 4
        print(f"  Chunk {i}: {'Header' if is_heading else 'Text'}")
        print(f"    Topic: {first_line}")
        print(f"    Length: {len(chunk.page_content)} chars")
    print()
    
    # Strategy 2: Balanced semantic chunks
    print("Strategy 2: Balanced Semantic Chunks")
    balanced_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=75,  # Higher overlap to preserve context
        separators=[
            "\n\n",       # Paragraph breaks (strong semantic boundary)
            "\n- ",       # List items
            ": ",         # Definitions/explanations
            ". ",         # Sentence boundaries
            " ",          # Word boundaries
        ]
    )
    
    balanced_chunks = balanced_splitter.split_documents([semantic_doc])
    print(f"Created {len(balanced_chunks)} balanced chunks")
    
    for i, chunk in enumerate(balanced_chunks[:3], 1):
        lines = chunk.page_content.strip().split('\n')
        first_line = lines[0] if lines else ""
        last_line = lines[-1] if len(lines) > 1 else ""
        
        print(f"  Chunk {i}:")
        print(f"    Starts: {first_line[:40]}...")
        print(f"    Ends: ...{last_line[-40:] if last_line else ''}")
        print(f"    Length: {len(chunk.page_content)} chars")
    print()

def demonstrate_content_type_optimization():
    """Show how to optimize splitting for different content types."""
    print("=== Content-Type Optimized Splitting ===\n")
    
    content_types = {
        "narrative": """Once upon a time, in a distant galaxy, there lived a brave explorer named Captain Sarah Chen. She had spent years traveling through space, discovering new planets and civilizations. Her ship, the Stellar Wanderer, was equipped with the most advanced technology of her era. One day, while exploring the outer rim of the galaxy, she detected an unusual signal coming from a small, uncharted planet.""",
        
        "technical": """API Rate Limiting Implementation: Configure rate limiters using sliding window algorithm. Set maximum requests per time window. Implement exponential backoff for exceeded limits. Monitor rate limit metrics: request count, time windows, rejected requests. Use Redis for distributed rate limiting across multiple servers.""",
        
        "conversational": """User: How do I reset my password?\nAgent: I can help you reset your password. Please click on 'Forgot Password' on the login page.\nUser: I don't see that option.\nAgent: The 'Forgot Password' link is located below the password field on our login page. It's in small blue text.\nUser: Found it! What happens next?\nAgent: You'll receive an email with reset instructions within 5 minutes. Check your spam folder if you don't see it."""
    }
    
    for content_type, content in content_types.items():
        print(f"{content_type.title()} Content:")
        
        doc = Document(page_content=content, metadata={"content_type": content_type})
        
        if content_type == "narrative":
            # For stories: preserve sentence flow, larger chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=30,
                separators=[". ", ", ", " ", ""]
            )
        elif content_type == "technical":
            # For technical docs: preserve complete concepts
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=150,
                chunk_overlap=20,
                separators=[": ", ". ", " ", ""]
            )
        else:  # conversational
            # For conversations: preserve turns
            splitter = CharacterTextSplitter(
                chunk_size=100,
                chunk_overlap=10,
                separator="\n"
            )
        
        chunks = splitter.split_documents([doc])
        print(f"  {len(chunks)} chunks optimized for {content_type} content")
        
        for i, chunk in enumerate(chunks[:2], 1):
            print(f"    Chunk {i}: {chunk.page_content[:50]}...")
        print()

if __name__ == "__main__":
    print("Advanced Text Splitting - Part 1d")
    print("Focus: Code-aware splitting and semantic boundary preservation\n")
    
    demonstrate_code_aware_splitting()
    # demonstrate_mixed_format_splitting()
    # demonstrate_semantic_boundary_preservation()
    # demonstrate_content_type_optimization()
    
    print("Key Takeaways:")
    print("Use custom separators for code and structured content")
    print("Preserve semantic boundaries with appropriate split points")
    print("Balance chunk size with context preservation")
    print("Optimize splitting strategy based on content type")
    print("Use higher overlap for complex documents to maintain context")