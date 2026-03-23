"""
Agent Coordinator - manages multiple ODA agents and resolves conflicts.

Uses collections.deque(maxlen=1000) for bounded conflict history.
"""

import asyncio
import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentState

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """
    Coordinates multiple ODA agents, handling start/stop lifecycle
    and resolving conflicting actions between agents.
    """

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        # Bounded conflict history
        self._conflict_history: deque = deque(maxlen=1000)
        self._running = False
        logger.info("AgentCoordinator initialized")

    def register(self, agent: BaseAgent):
        """Register an agent with the coordinator."""
        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    def unregister(self, name: str):
        """Unregister an agent and stop it if running."""
        if name in self._agents:
            self._agents[name].stop()
            if name in self._tasks:
                self._tasks[name].cancel()
                del self._tasks[name]
            del self._agents[name]
            logger.info(f"Unregistered agent: {name}")

    async def start_all(self):
        """Start all registered agents."""
        self._running = True
        for name, agent in self._agents.items():
            if agent.state != AgentState.RUNNING:
                task = asyncio.create_task(agent.start())
                self._tasks[name] = task
                logger.info(f"Started agent: {name}")

    async def stop_all(self):
        """Stop all running agents."""
        self._running = False
        for name, agent in self._agents.items():
            agent.stop()
        for name, task in self._tasks.items():
            task.cancel()
        self._tasks.clear()
        logger.info("All agents stopped")

    def pause_all(self):
        """Pause all running agents."""
        for agent in self._agents.values():
            agent.pause()

    def resume_all(self):
        """Resume all paused agents."""
        for agent in self._agents.values():
            agent.resume()

    def record_conflict(self, agent_a: str, agent_b: str,
                        channel: int, detail: str):
        """Record a conflict between two agents."""
        entry = {
            "time": time.time(),
            "agents": (agent_a, agent_b),
            "channel": channel,
            "detail": detail,
        }
        self._conflict_history.append(entry)
        logger.warning(f"Conflict: {agent_a} vs {agent_b} on ch{channel}: {detail}")

    def get_conflict_history(self, limit: int = 50) -> List[Dict]:
        """Return recent conflict entries."""
        return list(self._conflict_history)[-limit:]

    def get_status(self) -> Dict[str, Any]:
        """Return status of all agents."""
        return {
            "running": self._running,
            "agents": {name: agent.get_status()
                       for name, agent in self._agents.items()},
            "conflicts_total": len(self._conflict_history),
            "conflicts_recent": self.get_conflict_history(10),
        }
