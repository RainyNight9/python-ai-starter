.PHONY: backend frontend frontend-build test check

backend:
	python3 -m uvicorn app.main:app --reload

frontend:
	cd frontend && pnpm dev

frontend-build:
	cd frontend && pnpm build

test:
	python3 -m py_compile app/api/endpoints.py app/core/config.py app/rag/document_processor.py app/agent/tools.py app/agent/tool_definitions.py app/services/agent_service.py app/services/llm.py

check: test frontend-build
