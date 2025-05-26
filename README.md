# 5G Network Slicing with AI Integration

This project implements an AI-enhanced 5G network slicing system, focusing on the interaction between AI agents and existing network functions through Model Context Protocol (MCP).

## PoC Objectives

1. Study the use of AI pipelines to build and manage new AI technologies in the network
2. Demonstrate alternatives and trade-offs for interaction between AI agents and existing, non-AI Native network functions
3. Validate the hypothesis that using interface protocols (e.g., model context protocol) between agents and the rest of the network significantly reduces migration effort to AI Native networks

## Features

- Network simulation with dynamic slice allocation
- AI-driven traffic forecasting
- RAG (Retrieval Augmented Generation) for feature-specific inference
- MCP-based API integration
- Integration with ITU CG Datasets for traffic forecasting

## Project Structure

```
.
├── 5G-Network-Slicing/     # Core network simulation
│   ├── generate_data.py    # Data generation and simulation
│   └── slicesim/          # Simulation components
├── ai_agents/             # AI agent implementations
├── knowledge_base/        # RAG knowledge base
└── docs/                 # Documentation
```

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure environment variables
4. Run the simulation

## PoC Demonstration Scenarios

1. Time-series inference triggering MCP-based APIs to the orchestrator
2. Knowledge base techniques effectiveness in feature-specific inference

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 