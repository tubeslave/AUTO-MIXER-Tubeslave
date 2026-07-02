"""
Base Agent - ODA (Observe-Decide-Act) pattern abstract base class.

Provides dual execution support (async primary, sync fallback)
and correct state machine for RUNNING/PAUSED/STOPPED transitions.
"""

import asyncio
import enum
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AgentState(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class BaseAgent(ABC):
    """
    Abstract base agent implementing the ODA (Observe-Decide-Act) cycle.

    Subclasses must implement:
      - observe() -> Dict: gather current state
      - decide(observation: Dict) -> Dict: decide on action
      - act(decision: Dict): execute the action
    """

    def __init__(self, name: str, cycle_interval: float = 1.0):
        """
        Args:
            name: Human-readable agent name.
            cycle_interval: Seconds between ODA cycles.
        """
        self.name = name
        self.cycle_interval = cycle_interval
        self.state = AgentState.IDLE
        self._lock = threading.Lock()
        self._cycle_count = 0
        self._last_error: Optional[str] = None
        self._config: Dict[str, Any] = {}
        # Событие для PAUSED вместо busy-poll (правило CLAUDE.md: asyncio.Event.wait())
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        # Подряд идущие ошибки цикла; ERROR только после порога, чтобы
        # кратковременный обрыв связи с пультом не убивал агента навсегда
        self._consecutive_errors = 0
        self._max_consecutive_errors = 3
        logger.info(f"Agent '{name}' created (interval={cycle_interval}s)")

    # ------ Abstract ODA methods ------

    @abstractmethod
    async def observe(self) -> Dict[str, Any]:
        """Gather observations from the environment."""
        ...

    @abstractmethod
    async def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Decide what action to take based on observations."""
        ...

    @abstractmethod
    async def act(self, decision: Dict[str, Any]) -> None:
        """Execute the decided action."""
        ...

    # ------ Lifecycle ------

    async def start(self) -> None:
        """Start the agent's ODA loop."""
        if self.state == AgentState.RUNNING:
            logger.warning(f"Agent '{self.name}' already running")
            return
        if self.state == AgentState.PAUSED:
            # Старый _run_loop жив (он не выходит при PAUSED) — второй цикл
            # слал бы дублирующиеся OSC-записи на пульт. Только возобновляем.
            self.resume()
            return
        self.state = AgentState.RUNNING
        logger.info(f"Agent '{self.name}' started")
        await self._run_loop()

    async def _run_loop(self) -> None:
        """Main ODA loop. Continues while RUNNING or PAUSED."""
        while self.state in (AgentState.RUNNING, AgentState.PAUSED):
            # If paused, wait without doing work (без busy-poll)
            if self.state == AgentState.PAUSED:
                await self._resume_event.wait()
                continue

            try:
                observation = await self.observe()
                decision = await self.decide(observation)
                if decision:
                    await self.act(decision)
                self._cycle_count += 1
                self._consecutive_errors = 0
            except Exception as e:
                self._last_error = str(e)
                self._consecutive_errors += 1
                if self._consecutive_errors >= self._max_consecutive_errors:
                    logger.error(
                        f"Agent '{self.name}' cycle error ({self._consecutive_errors} "
                        f"consecutive), entering ERROR state: {e}"
                    )
                    self.state = AgentState.ERROR
                    break
                logger.warning(
                    f"Agent '{self.name}' cycle error "
                    f"({self._consecutive_errors}/{self._max_consecutive_errors}), "
                    f"retrying: {e}"
                )

            await asyncio.sleep(self.cycle_interval)

        logger.info(f"Agent '{self.name}' loop exited (state={self.state.value})")

    def pause(self) -> None:
        """Pause the agent. Can be resumed."""
        if self.state == AgentState.RUNNING:
            self.state = AgentState.PAUSED
            self._resume_event.clear()
            logger.info(f"Agent '{self.name}' paused")

    def resume(self) -> None:
        """Resume a paused agent."""
        if self.state == AgentState.PAUSED:
            self.state = AgentState.RUNNING
            self._resume_event.set()
            logger.info(f"Agent '{self.name}' resumed")

    def stop(self) -> None:
        """Stop the agent permanently."""
        self.state = AgentState.STOPPED
        # Будим цикл, если он ждёт resume в PAUSED, чтобы он мог выйти
        self._resume_event.set()
        logger.info(f"Agent '{self.name}' stopped")

    def configure(self, config: Dict[str, Any]) -> None:
        """Update agent configuration."""
        self._config.update(config)
        logger.debug(f"Agent '{self.name}' configured: {list(config.keys())}")

    def get_status(self) -> Dict[str, Any]:
        """Return agent status summary."""
        return {
            "name": self.name,
            "state": self.state.value,
            "cycles": self._cycle_count,
            "last_error": self._last_error,
            "interval": self.cycle_interval,
        }
