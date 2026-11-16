"""PhoneAgent: one Agent class for both directions, with every before/after hook in one place.

Customer-specific behavior comes from CallConfig, not from subclasses: Strada runs any
VAPI assistant through one agent, and Stuut runs every org through one OutboundCallAgent.

Hooks, in the order they run on a call:

  on_enter                 BEFORE the first turn: this agent just took over the session
  on_user_turn_completed   BEFORE each LLM reply: the user's turn is final, the LLM has not run
  llm_node                 AROUND the LLM: wraps the model's output stream
  tts_node                 BEFORE speech: rewrites text on its way to TTS
  tools                    DURING: the LLM calls these
  on_exit                  AFTER: this agent is handing off or the session is closing
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import AsyncIterable

import aiohttp
from livekit import rtc
from livekit.agents import Agent, ModelSettings, RunContext, ToolError, function_tool, llm
from livekit.agents.voice.speech_handle import SpeechHandle

from shared.contract import CallConfig

from .state import CallState
from .telephony import hang_up, press_keys

logger = logging.getLogger("field-patterns.agent")

OPT_OUT = re.compile(r"\b(stop calling|do not call|don't call|take me off|remove me)\b", re.IGNORECASE)
DIGITS = re.compile(r"^[0-9*#]{1,16}$")

# Stand-in results for tools when config.mock is set (console, tests, simulations).
MOCK_TOOL_RESULTS = {
    "lookup_invoice": lambda args: {
        "invoice_number": args["invoice_number"],
        "status": "open",
        "amount_due": "$1,240.00",
        "due_date": "October 31",
    },
}


def apply_pronunciations(text: str, pronunciations: dict[str, str]) -> str:
    """Replace words the voice says wrong (Strada: VAPI formatPlan replacements)."""
    for written, spoken in pronunciations.items():
        text = text.replace(written, spoken)
    return text


class PhoneAgent(Agent):
    def __init__(
        self,
        config: CallConfig,
        *,
        greet_on_enter: bool,
        chat_ctx: llm.ChatContext | None = None,
    ) -> None:
        super().__init__(instructions=config.instructions, chat_ctx=chat_ctx)
        self._config = config
        self._greet_on_enter = greet_on_enter

    # ------------------------------------------------------------------ BEFORE

    async def on_enter(self) -> None:
        """Inbound greets here. Outbound waits: the AMD verdict decides (agent/outbound.py)."""
        if self._greet_on_enter:
            self.greet()

    def greet(self) -> SpeechHandle:
        if self._config.greeting:
            return self.session.say(self._config.greeting)
        return self.session.generate_reply(instructions="Greet the caller in one short sentence.")

    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        """Deterministic checks before the model answers.

        Changes to turn_ctx apply to this reply only. Rules that must not depend on the
        model (here: a do-not-call request) belong in code, not only in the prompt.
        """
        text = new_message.text_content or ""
        if OPT_OUT.search(text):
            call: CallState = self.session.userdata
            call.end_with("caller-opted-out")
            turn_ctx.add_message(
                role="system",
                content=(
                    "The caller asked not to be called. Apologize, confirm they will be removed "
                    "from the list, say goodbye, and call end_call."
                ),
            )

    # ------------------------------------------------------------------ AROUND THE MODEL

    async def llm_node(
        self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
    ) -> AsyncIterable[llm.ChatChunk | str]:
        """Wrap the default LLM node. Here: record time to first token.

        Stuut uses the same override to strip tool-call syntax the model writes as text.
        """
        started = time.perf_counter()
        first = True
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            if first:
                call: CallState = self.session.userdata
                call.llm_first_token_s.append(round(time.perf_counter() - started, 3))
                first = False
            yield chunk

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings) -> AsyncIterable[rtc.AudioFrame]:
        """Rewrite text before TTS. Also covers session.say(), which skips the LLM.

        Chunks can split a word, so a production version buffers to a punctuation
        boundary first (Strada: pipeline/text-transforms.ts).
        """
        pronunciations = self._config.pronunciations

        async def rewritten() -> AsyncIterable[str]:
            async for chunk in text:
                yield apply_pronunciations(chunk, pronunciations)

        async for frame in Agent.default.tts_node(self, rewritten(), model_settings):
            yield frame

    # ------------------------------------------------------------------ DURING: tools

    @function_tool
    async def lookup_invoice(self, ctx: RunContext[CallState], invoice_number: str) -> str:
        """Look up an invoice's status, amount due and due date.

        Args:
            invoice_number: The invoice number the caller gave, for example INV-20417.
        """
        # Strada proxies every tool to the customer's HTTP tool server and holds the
        # shared secret in the worker, never in dispatch metadata.
        config = ctx.userdata.config
        arguments = {"invoice_number": invoice_number}
        if config.mock or not config.tool_server_url:
            return json.dumps(MOCK_TOOL_RESULTS["lookup_invoice"](arguments))
        async with aiohttp.ClientSession() as http:
            async with http.post(
                config.tool_server_url,
                json={"call_id": config.call_id, "tool": "lookup_invoice", "arguments": arguments},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    raise ToolError(f"invoice lookup failed with status {resp.status}")
                return await resp.text()

    @function_tool
    async def send_dtmf(self, ctx: RunContext[CallState], digits: str) -> str:
        """Press keys on a phone menu. Use only keys the menu announced, once, then wait.

        Args:
            digits: Keys to press, using 0-9, * and #.
        """
        if not DIGITS.match(digits):
            return "Not sent: digits must be 0-9, * or #."
        ctx.userdata.in_phone_menu = True
        if ctx.userdata.config.mock:
            return f"(mock) pressed {digits}"
        await press_keys(digits)
        return f"Pressed {digits}."

    @function_tool
    async def end_call(self, ctx: RunContext[CallState], said_goodbye: bool = False) -> None:
        """End the call when the conversation is finished or the caller wants to hang up.

        Do not end the call while the caller has an open question or asked you to wait.

        Args:
            said_goodbye: True only if your reply in this same turn already says goodbye.
        """
        # Let the current reply finish first (Stuut: agents/call.py end_call).
        await ctx.wait_for_playout()
        if not said_goodbye:
            await ctx.session.say("Thanks for your time. Goodbye!", allow_interruptions=False)
        ctx.userdata.end_with("agent-ended-call")
        await hang_up("agent-ended-call")
        # Returning nothing keeps the model from speaking again after the hang-up.

    # ------------------------------------------------------------------ AFTER

    async def on_exit(self) -> None:
        """Runs when this agent hands off to another one or the session closes."""
        call: CallState = self.session.userdata
        logger.info(
            "agent exiting",
            extra={"call_id": call.config.call_id, "llm_first_token_s": call.llm_first_token_s},
        )
