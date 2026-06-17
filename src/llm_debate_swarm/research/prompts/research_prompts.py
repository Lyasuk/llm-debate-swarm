"""Research prompt templates for various data sources."""

GENERATE_SEARCH_QUERIES = """\
Given this prediction market question, generate 3 diverse web search queries
to gather comprehensive research data.

Question: {question}
Category: {category}

Requirements:
- Query 1: Direct factual search about the topic
- Query 2: Expert analysis and prediction search
- Query 3: Recent news and developments search

Output 3 queries, one per line.
"""
