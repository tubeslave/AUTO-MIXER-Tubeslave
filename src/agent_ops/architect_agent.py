class ArchitectAgent:
    def __init__(self, name: str = "architect") -> None:
        self.name = name

    def run(self, project_state):
        return {
            "agent": self.name,
            "status": "ok",
            "result": {
                "summary": "Architecture review placeholder",
                "next_recommendation": "Check dataset schema, training entrypoint, and evaluation interfaces.",
            },
        }
