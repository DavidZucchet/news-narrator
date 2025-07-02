# MCP-Integrated News Narrator System

A system that fetches live news headlines using MCP (Model Context Protocol) servers, generates AI-powered images for each headline, and provides an interactive narration experience with pause/resume functionality and intelligent Q&A capabilities.

## 🚀 Features

- **Live News Fetching**: Uses MCP servers to fetch real-time headlines from major news sources
- **AI Image Generation**: Creates relevant images for each headline using Stable Diffusion
- **Interactive Narration**: Pause/resume functionality with real-time user interaction
- **Intelligent Q&A**: Ask questions about headlines and get AI-powered answers
- **Graceful Fallbacks**: Works even when MCP server fails with sample headlines

## 📋 Prerequisites

- Python 3.12+
- Node.js (for MCP server)
- OpenAI API key
- uv package manager

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd news-narrator
   ```

2. **Install dependencies with uv**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install project dependencies from pyproject.toml
   uv sync
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file manually with your OpenAI API key
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

4. **Install Playwright browsers (if needed)**
   ```bash
   # If you don't have Chrome or Chromium installed
   uv run playwright install chromium
   ```

## 🚀 Usage

### Run the application
```bash
uv run python main.py
```

**⏰ First Run Note:** The initial execution will take approximately **10 minutes** to download the Stable Diffusion model (~4GB). Subsequent runs will start much faster as the model is cached locally.

### Interactive Flow
1. **Choose headline count**: Enter number of headlines (1-20, default: 5)
2. **Wait for fetch**: System fetches live headlines using MCP
3. **Review headlines**: See the list of headlines to be narrated
4. **Interactive control**:
   - Press `Enter` to pause/resume narration
   - Type questions to get AI-powered answers
   - Images are automatically generated for each headline

### Example Session
```
📊 How many headlines would you like? (1-20, default: 5): 3
📰 Fetching 3 latest headlines...
✅ Successfully fetched 3 headlines!

🟢 Narration starting. Press ENTER to pause/resume or type questions.

📰 Headline 1: Major breakthrough in renewable energy storage announced
🛠️ Generating image...
🖼️ Image saved at: generated_images/major_breakthrough_renewable_energy.png

[Press Enter to pause, type questions, or let it continue...]
```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `LOG_LEVEL`: Logging level (default: INFO)
- `IMAGE_OUTPUT_DIR`: Directory for generated images (default: generated_images)
- `MCP_TIMEOUT`: MCP server timeout in seconds (default: 60)

## 🧪 Testing

### Test headline sources
```bash
uv run python -c "
import asyncio
from main import fetch_latest_headlines

async def test_headlines():
    print('Testing headline fetching...')
    try:
        headlines = await fetch_latest_headlines(3)
        print(f'✅ Fetched {len(headlines)} headlines:')
        for i, h in enumerate(headlines, 1):
            print(f'   {i}. {h}')
    except Exception as e:
        print(f'❌ Error: {e}')

asyncio.run(test_headlines())
"
```

### Run with development dependencies
```bash
# Install development dependencies
uv sync --group dev

# Run linting and formatting
uv run black main.py
uv run flake8 main.py
uv run mypy main.py
```

## 🐛 Troubleshooting

### Common Issues

**MCP Server Connection Failed**
- Ensure Node.js is installed: `node --version`
- If you get browser-related errors, install Playwright browsers: `uv run playwright install chromium`
- Check network connectivity

**Image Generation Errors**
- **First run takes time**: Initial setup downloads Stable Diffusion model (~4GB, ~10 minutes)
- **Subsequent runs**: Much faster as model is cached locally
- The system uses CPU by default with optimizations
- Check disk space in output directory
- For faster generation, ensure you have adequate RAM

**OpenAI API Errors**
- Verify API key in `.env` file
- Check API quota and billing
- Ensure stable internet connection

**Browser/Playwright Issues**
- Install required browsers: `uv run playwright install chromium`
- On some systems, you might need: `uv run playwright install chrome`

### Debug Mode
Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
uv run python main.py
```

## 📁 Project Structure

```
news-narrator/
├── main.py                    # Final polished version with all features
├── FullMCPNarratorImage.py   # Full integration (development step 4)
├── NarratorImage.py          # Basic narrator with Q&A (development step 3)
├── image.py                  # Image generation component (development step 2)
├── MCP.py                    # MCP integration component (development step 1)
├── pyproject.toml            # Project configuration with dependencies
├── .env                      # Environment variables (create manually)
├── README.md                 # This file
└── generated_images/         # Generated images (created at runtime)
```

## 🚀 Development History

This project was built incrementally, demonstrating a systematic approach to tackling complex problems:

1. **MCP.py** - Started with basic MCP server integration for web browsing
2. **image.py** - Added Stable Diffusion image generation capability  
3. **NarratorImage.py** - Created basic narrator system with Q&A functionality
4. **FullMCPNarratorImage.py** - Integrated MCP headline fetching with narrator
5. **main.py** - Final polished version with improved UX and error handling

Each file represents a working milestone in the development process, showing how the project evolved from individual components to a complete integrated system.

## 🔧 Development

### Code Style
```bash
# Format code
uv run black main.py

# Type checking  
uv run mypy main.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with proper tests
4. Ensure code quality: `black`, `flake8`, `mypy`
5. Commit with descriptive messages
6. Push and create a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenAI for GPT models and API
- Stability AI for Stable Diffusion
- MCP (Model Context Protocol) project
- uv package manager by Astral

---

**Version**: 1.0.0  
**Python Version**: 3.12+  
**Package Manager**: uv