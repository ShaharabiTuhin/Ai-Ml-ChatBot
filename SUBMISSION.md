# Final Submission Package

Use this file to finalize all required submission artifacts.

## 1) GitHub Repository
- URL: <ADD_GITHUB_REPO_LINK>
- Includes source code, README, and dependency/setup steps

## 2) YouTube Video Presentation
- URL: <ADD_YOUTUBE_VIDEO_LINK>
- Language: English
- Must include:
  - Part 1: Full project demo (implemented agents/features)
  - Part 2: Codebase walkthrough (structure, key functions, tools, RAG, tracing)

## 3) LangSmith Tracing Evidence
- Project Name: <ADD_LANGSMITH_PROJECT_NAME>
- Trace Link 1 (RAG request): <ADD_TRACE_LINK>
- Trace Link 2 (OCR request): <ADD_TRACE_LINK>
- Trace Link 3 (Web/Google grounded search request): <ADD_TRACE_LINK>

## Live Demo Checklist
- [ ] Chat message flow works
- [ ] Session history loads
- [ ] RAG answers use knowledge base
- [ ] OCR image upload returns extracted text
- [ ] Internet search answers include references
- [ ] Google grounded search works when CSE keys are configured
- [ ] LangSmith shows chain/tool/retriever traces

## Final Packaging Checklist
- [ ] `.env` is not committed
- [ ] `.env.example` is complete
- [ ] `requirements.txt` is up to date
- [ ] `README.md` has setup + run instructions
- [ ] FAISS index can be rebuilt via `Database/setup_faiss.py`
