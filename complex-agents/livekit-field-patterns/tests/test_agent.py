"""Agent wiring and decisions that can be checked without a room or a model."""

from livekit.agents import AMDCategory

import agent.outbound as outbound
from agent.main import load_call_config
from agent.phone_agent import OPT_OUT, PhoneAgent, apply_pronunciations
from agent.state import CallState
from shared.samples import sample_inbound_config, sample_outbound_config


def test_phone_agent_exposes_its_tools_and_prompt():
    config = sample_inbound_config()
    agent = PhoneAgent(config, greet_on_enter=True)

    assert {tool.id for tool in agent.tools} == {"lookup_invoice", "send_dtmf", "end_call"}
    assert agent.instructions == config.instructions


def test_pronunciations_and_opt_out_rule():
    assert apply_pronunciations("Northwind INV-7", {"Northwind": "North wind", "INV-": "invoice "}) == (
        "North wind invoice 7"
    )
    assert OPT_OUT.search("Please take me off your list")
    assert not OPT_OUT.search("Can you call me back tomorrow?")


class _FakeJob:
    """Just enough of JobContext for load_call_config."""

    def __init__(self, metadata: str = "") -> None:
        self.job = type("Job", (), {"metadata": metadata})()


def test_entrypoint_uses_dispatch_metadata_or_falls_back_to_a_mock_sample(monkeypatch):
    real = sample_outbound_config(mock=False)
    assert load_call_config(_FakeJob(real.model_dump_json())) == real  # type: ignore[arg-type]

    monkeypatch.setenv("CONSOLE_DIRECTION", "outbound")
    fallback = load_call_config(_FakeJob(""))  # type: ignore[arg-type]
    assert fallback.direction == "outbound"
    assert fallback.mock is True


class _FakeSession:
    def __init__(self) -> None:
        self.said: list[str] = []
        self.replies = 0

    async def say(self, text: str, **_kwargs) -> None:
        self.said.append(text)

    def generate_reply(self, **_kwargs) -> None:
        self.replies += 1


class _FakeAgent:
    def __init__(self) -> None:
        self.greeted = False

    def greet(self) -> None:
        self.greeted = True


async def _verdict(category: AMDCategory, monkeypatch) -> tuple[CallState, _FakeSession, _FakeAgent, list[str]]:
    hung_up: list[str] = []

    async def fake_hang_up(reason: str) -> None:
        hung_up.append(reason)

    monkeypatch.setattr(outbound, "hang_up", fake_hang_up)
    call = CallState(config=sample_outbound_config())
    session, agent = _FakeSession(), _FakeAgent()
    await outbound.act_on_amd_verdict(category, session, agent, call)  # type: ignore[arg-type]
    return call, session, agent, hung_up


async def test_amd_human_gets_the_greeting(monkeypatch):
    call, session, agent, hung_up = await _verdict(AMDCategory.HUMAN, monkeypatch)
    assert agent.greeted and not hung_up


async def test_amd_voicemail_leaves_the_message_then_hangs_up(monkeypatch):
    call, session, agent, hung_up = await _verdict(AMDCategory.MACHINE_VM, monkeypatch)
    assert session.said == [call.config.voicemail_message]
    assert hung_up == ["left-voicemail"]
    assert call.ended_reason == "left-voicemail"


async def test_amd_phone_menu_stays_quiet_and_lets_the_model_navigate(monkeypatch):
    call, session, agent, hung_up = await _verdict(AMDCategory.MACHINE_IVR, monkeypatch)
    assert call.in_phone_menu
    assert session.replies == 1 and not session.said and not agent.greeted


async def test_amd_full_mailbox_hangs_up_silently(monkeypatch):
    call, session, agent, hung_up = await _verdict(AMDCategory.MACHINE_UNAVAILABLE, monkeypatch)
    assert hung_up == ["mailbox-unavailable"] and not session.said
