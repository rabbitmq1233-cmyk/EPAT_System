"""Broker registry: look up an adapter by name."""

from __future__ import annotations

from collections.abc import Callable

from epat.brokers.base import Broker, BrokerError
from epat.brokers.mock import MockBroker


def _zerodha(**kwargs) -> Broker:
    from epat.brokers.zerodha import ZerodhaBroker

    return ZerodhaBroker(**kwargs)


def _upstox(**kwargs) -> Broker:
    from epat.brokers.upstox import UpstoxBroker

    return UpstoxBroker(**kwargs)


def _ib(**kwargs) -> Broker:
    from epat.brokers.ib import IBBroker

    return IBBroker(**kwargs)


_FACTORIES: dict[str, Callable[..., Broker]] = {
    "mock": MockBroker,
    "zerodha": _zerodha,
    "upstox": _upstox,
    "ib": _ib,
}


def list_brokers() -> list[str]:
    """Names of all registered broker adapters."""
    return sorted(_FACTORIES)


def get_broker(name: str, **kwargs) -> Broker:
    """Instantiate a broker adapter by name (``mock``, ``zerodha``, ``upstox``, ``ib``)."""
    key = name.strip().lower()
    if key not in _FACTORIES:
        raise BrokerError(
            f"unknown broker '{name}'. Available: {', '.join(list_brokers())}"
        )
    return _FACTORIES[key](**kwargs)


def register_broker(name: str, factory: Callable[..., Broker]) -> None:
    """Register a custom broker factory."""
    _FACTORIES[name.strip().lower()] = factory
