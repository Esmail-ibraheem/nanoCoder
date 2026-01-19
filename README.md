  <div align="center">
  <h1>nanoCoder</h1>
  <p><strong>Your specialized AI assistant for deep coding tasks.</strong></p>

  [![Docker Hub](https://img.shields.io/docker/pulls/esmailg/nanocoder?logo=docker&style=flat-square)](https://hub.docker.com/repository/docker/esmailg/nanocoder/general)
   [![License](https://img.shields.io/github/license/Esmail-ibraheem/nanoCoder?style=flat-square)](https://github.com/Esmail-ibraheem/nanoCoder/tree/main?tab=MIT-1-ov-file#)
  [![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&style=flat-square)](https://www.python.org/)
</div>

<br />

`nanoCoder` is a powerful agentic coding assistant designed to help you solve complex coding tasks, manage projects, and automate workflows directly from your terminal. Whether you're building a new project or debugging legacy code, nanoCoder has the tools and intelligence to assist you at every step.

---

<img width="865" height="515" alt="nanoCoder in action" src="https://github.com/user-attachments/assets/6e7c7ebb-0145-49c8-82be-6892539cfc74" />

## ✨ Key Features

- 🧠 **Project Intelligence**: Automatically understands and navigates your codebase.
- 🛠 **Autonomous Tooling**: Built-in capabilities to read, write, grep, and execute shell commands.
- 🔗 **MCP Native**: Seamless integration with the Model Context Protocol (MCP) for extensible capabilities.
- 🎨 **Premium TUI**: A beautiful and responsive terminal interface for an immersive developer experience.
- 🐳 **Docker Ready**: Fully containerized for portability and consistent performance.
- 🚀 **OpenRouter Optimized**: Designed to leverage high-performance models via OpenRouter.

<img width="943" height="860" alt="Features visualization" src="https://github.com/user-attachments/assets/af779f94-4f2e-457e-90d1-20c290513cc2" />

## 🚀 Getting Started

### 📦 Installation

#### Local Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Esmail-ibraheem/nanoCoder.git
   cd nanoCoder
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. run main file
   ```bash
   python main.py
   ```

#### Docker Setup (Recommended)
<img width="612" height="408" alt="ChatGPT_Image_Jan_19__2026__04_12_17_PM-removebg-preview" src="https://github.com/user-attachments/assets/148e1f88-839e-4004-a489-e4b1d1a0cf21" />

**Run from Docker Hub:**
```bash
docker pull esmailg/nanocoder:latest
docker run -it -e API_KEY="your_api_key" esmailg/nanocoder:latest
```

**Build from Source:**
1. **Build the image**:
   ```bash
   docker build -t nanocoder .
   ```
2. **Run the image**:
   - **Basic run**
     ```bash
     docker run -it -e API_KEY="your_api_key" -e BASE_URL="https://openrouter.ai/api/v1" nanocoder
     ```
   - With File Access
To let the agent read/write files in your current directory:

   ```bash
   docker run -it \
    -v "$(pwd):/app/workspace" \
    -e API_KEY="$API_KEY" \
    -e BASE_URL="$BASE_URL" \
    nanocoder
   ```

<img width="900" height="499" alt="image" src="https://github.com/user-attachments/assets/f3e072b9-f3e5-492b-8ec4-df656eec615a" />


### ⚙️ Configuration

Set your environment variables to authenticate with your preferred LLM provider (OpenRouter recommended):

**Windows (PowerShell):**
```powershell
$env:API_KEY = "sk-or-v1-..."
$env:BASE_URL = "https://openrouter.ai/api/v1"
```

**Linux/Mac (Bash):**
```bash
export API_KEY="sk-or-v1-..."
export BASE_URL="https://openrouter.ai/api/v1"
```

---

<img width="804" height="428" alt="UI Showcase" src="https://github.com/user-attachments/assets/d8b602b8-dac9-45a4-8a72-ecb938232864" />



## 🤝 Contribution

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

---





