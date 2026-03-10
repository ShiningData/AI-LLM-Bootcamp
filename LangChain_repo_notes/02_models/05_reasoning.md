## Reasoning
Newer models are capable of performing multi-step reasoning to arrive at a conclusion. This involves breaking down complex problems into smaller, more manageable steps.
If supported by the underlying model, you can surface this reasoning process to better understand how the model arrived at its final answer.

- main9_reasoning.py
```python
for chunk in model.stream("Why do parrots have colorful feathers?"):
    reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
    print(reasoning_steps if reasoning_steps else chunk.text)
```

- Run
```bash
python main9_reasoning.py
```