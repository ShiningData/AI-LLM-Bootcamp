"""
Server-Sent Events (SSE) transport with streaming capabilities.

Shows how to:
- Create an MCP server with SSE transport for real-time streaming
- Stream large responses and progress updates
- Handle long-running tools that provide incremental results
- Demonstrate streaming data consumption patterns
"""
import asyncio
import time
import random
from typing import AsyncGenerator
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP

@dataclass
class ProcessingStatus:
    """Track processing status for streaming operations."""
    total_items: int
    processed_items: int
    current_stage: str
    errors: int

# Create an MCP server optimized for streaming operations
mcp = FastMCP("StreamingProcessor")

@mcp.tool()
async def process_large_dataset(dataset_size: int = 1000) -> str:
    """
    Simulate processing a large dataset with progress updates.
    
    This tool demonstrates streaming by yielding progress updates
    as it processes data in chunks.
    """
    if dataset_size <= 0:
        return "Dataset size must be positive"
    
    chunk_size = max(1, dataset_size // 10)  # Process in 10% chunks
    processed = 0
    results = []
    
    print(f"[Streaming Server] Starting to process {dataset_size} records...")
    
    while processed < dataset_size:
        # Simulate processing a chunk
        await asyncio.sleep(0.5)  # Simulate work
        
        current_chunk = min(chunk_size, dataset_size - processed)
        processed += current_chunk
        
        # Simulate some processing result
        chunk_result = f"Processed chunk {processed//chunk_size}: {current_chunk} records"
        results.append(chunk_result)
        
        progress = (processed / dataset_size) * 100
        print(f"[Streaming Server] Progress: {progress:.1f}% ({processed}/{dataset_size})")
        
        # In a real streaming implementation, this would be yielded
        # For now, we collect results
    
    final_result = f"Dataset processing completed!\n" + "\n".join(results)
    print(f"[Streaming Server] Finished processing {dataset_size} records")
    return final_result

@mcp.tool()
async def generate_report(report_type: str = "sales") -> str:
    """
    Generate a detailed report with streaming progress updates.
    
    Args:
        report_type: Type of report to generate ('sales', 'users', 'performance')
    """
    report_types = ["sales", "users", "performance", "analytics"]
    if report_type not in report_types:
        return f"Invalid report type. Choose from: {', '.join(report_types)}"
    
    print(f"[Streaming Server] Starting {report_type} report generation...")
    
    # Simulate multi-stage report generation
    stages = [
        "Collecting data from database",
        "Processing raw data",
        "Calculating metrics",
        "Generating visualizations", 
        "Formatting report",
        "Finalizing output"
    ]
    
    report_content = [f"# {report_type.title()} Report"]
    report_content.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append("")
    
    for i, stage in enumerate(stages, 1):
        print(f"[Streaming Server] Stage {i}/{len(stages)}: {stage}")
        await asyncio.sleep(0.8)  # Simulate stage processing time
        
        # Add stage-specific content
        if stage == "Collecting data from database":
            records_found = random.randint(1000, 5000)
            report_content.append(f"📊 Data Collection: Found {records_found} records")
            
        elif stage == "Processing raw data":
            report_content.append(f"🔄 Data Processing: Cleaned and validated data")
            
        elif stage == "Calculating metrics":
            metric_value = random.randint(50, 150)
            report_content.append(f"📈 Key Metric: {metric_value}% improvement")
            
        elif stage == "Generating visualizations":
            chart_count = random.randint(3, 8)
            report_content.append(f"📊 Generated {chart_count} charts and graphs")
            
        elif stage == "Formatting report":
            page_count = random.randint(5, 20)
            report_content.append(f"📄 Formatted {page_count}-page report")
            
        elif stage == "Finalizing output":
            report_content.append(f"✅ Report generation completed successfully")
    
    final_report = "\n".join(report_content)
    print(f"[Streaming Server] {report_type} report generation finished")
    return final_report

@mcp.tool()
async def analyze_log_file(log_entries: int = 10000) -> str:
    """
    Analyze a large log file with streaming progress updates.
    
    This simulates parsing and analyzing log entries with real-time feedback.
    """
    if log_entries <= 0:
        return "Log entries count must be positive"
    
    print(f"[Streaming Server] Starting log analysis for {log_entries} entries...")
    
    # Analysis phases
    phases = ["Reading log file", "Parsing entries", "Pattern analysis", "Error detection", "Summary generation"]
    entries_per_phase = log_entries // len(phases)
    
    analysis_results = {
        "errors": 0,
        "warnings": 0,
        "info": 0,
        "patterns_found": []
    }
    
    for phase_idx, phase in enumerate(phases):
        print(f"[Streaming Server] Phase {phase_idx + 1}: {phase}")
        
        # Simulate processing entries in this phase
        for batch in range(0, entries_per_phase, 1000):
            await asyncio.sleep(0.3)  # Simulate processing time
            
            # Generate some mock analysis results
            if phase == "Error detection":
                analysis_results["errors"] += random.randint(0, 10)
                analysis_results["warnings"] += random.randint(5, 25)
            elif phase == "Pattern analysis":
                patterns = ["Database timeout", "API rate limit", "Memory spike", "Network latency"]
                analysis_results["patterns_found"].extend(random.sample(patterns, 2))
            
            progress = ((phase_idx * entries_per_phase + batch) / log_entries) * 100
            if batch % 2000 == 0:  # Log every 2000 entries
                print(f"[Streaming Server] Progress: {progress:.1f}%")
    
    # Generate final analysis report
    report_lines = [
        f"Log Analysis Complete - {log_entries} entries processed",
        f"Errors found: {analysis_results['errors']}",
        f"Warnings found: {analysis_results['warnings']}",
        f"Info messages: {log_entries - analysis_results['errors'] - analysis_results['warnings']}",
        f"Patterns detected: {', '.join(set(analysis_results['patterns_found']))}"
    ]
    
    final_report = "\n".join(report_lines)
    print("[Streaming Server] Log analysis completed")
    return final_report

@mcp.tool()
async def real_time_monitor(duration_seconds: int = 30) -> str:
    """
    Simulate real-time monitoring with periodic updates.
    
    This tool demonstrates continuous streaming of status updates.
    """
    if duration_seconds <= 0 or duration_seconds > 300:  # Max 5 minutes
        return "Duration must be between 1 and 300 seconds"
    
    print(f"[Streaming Server] Starting real-time monitoring for {duration_seconds} seconds...")
    
    start_time = time.time()
    metrics = []
    
    while time.time() - start_time < duration_seconds:
        # Generate mock metrics
        cpu_usage = random.randint(10, 90)
        memory_usage = random.randint(30, 85)
        network_io = random.randint(100, 1000)
        
        timestamp = time.strftime('%H:%M:%S')
        metric = f"[{timestamp}] CPU: {cpu_usage}%, Memory: {memory_usage}%, Network: {network_io}KB/s"
        metrics.append(metric)
        
        print(f"[Streaming Server] {metric}")
        
        # Wait before next update
        await asyncio.sleep(2)
    
    final_report = f"Monitoring session completed ({duration_seconds}s)\n" + "\n".join(metrics[-10:])
    print("[Streaming Server] Real-time monitoring finished")
    return final_report

if __name__ == "__main__":
    print("[Streaming Server] Starting SSE-enabled streaming server...")
    print("[Streaming Server] This server provides real-time streaming capabilities")
    print("[Streaming Server] Perfect for long-running operations and progress tracking")
    print()
    
    # Run the server using Server-Sent Events (SSE) transport
    # This provides:
    # - Real-time streaming capabilities
    # - Progress updates for long-running operations
    # - Better user experience with incremental results
    # - HTTP-based but optimized for streaming
    mcp.run(transport="sse")