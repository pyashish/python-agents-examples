"""Sample calls used by the walkthrough, the tests and console mode.

Console runs and `lk agent simulate` have no dispatch metadata, so both customers
fall back to a bundled sample call (Strada: config/fallback-job-metadata.json,
Stuut: MockOutboundCallJob). These are ours.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from .contract import CallConfig


@dataclass
class Customer:
    """What the backend knows about who is being called. Never sent whole to the agent."""

    name: str
    company: str
    invoice_number: str
    amount_due: str
    due_date: str


SAMPLE_CUSTOMER = Customer(
    name="Dana Ortiz",
    company="Ortiz Landscaping",
    invoice_number="INV-20417",
    amount_due="$1,240.00",
    due_date="October 31",
)


def inbound_instructions(business: str) -> str:
    """Strada's prompt is the VAPI assistant's system message, rendered with call variables."""
    return (
        f"You answer the phone for {business}. Help callers check an invoice or report a "
        "billing problem. Keep replies to one or two short sentences; this is a phone call. "
        "Use lookup_invoice when the caller gives an invoice number. When the caller is done, "
        "say goodbye and call end_call."
    )


def outbound_instructions(customer: Customer, business: str) -> str:
    """Stuut builds the full prompt in the backend (build_prompt) before dispatching."""
    return (
        f"You are calling {customer.name} at {customer.company} on behalf of {business} "
        f"about invoice {customer.invoice_number} for {customer.amount_due}, due "
        f"{customer.due_date}. Confirm you are speaking with the right person, explain why "
        "you are calling, and ask when they expect to pay. If you reach a phone menu, use "
        "send_dtmf only with keys the menu announced. Keep replies short. When the "
        "conversation is finished, call end_call."
    )


def sample_inbound_config(*, mock: bool = True) -> CallConfig:
    return CallConfig(
        call_id=str(uuid.uuid4()),
        direction="inbound",
        instructions=inbound_instructions("Northwind Supply"),
        greeting="Thanks for calling Northwind Supply billing. How can I help?",
        pronunciations={"Northwind": "North wind"},
        mock=mock,
    )


def sample_outbound_config(*, mock: bool = True) -> CallConfig:
    customer = SAMPLE_CUSTOMER
    return CallConfig(
        call_id=str(uuid.uuid4()),
        direction="outbound",
        instructions=outbound_instructions(customer, "Northwind Supply"),
        greeting=f"Hi, this is Northwind Supply calling for {customer.name}. Is now a good time?",
        voicemail_message=(
            f"Hi {customer.name}, this is Northwind Supply about invoice "
            f"{customer.invoice_number}. Please call us back at your convenience. Thank you."
        ),
        to_number="+15555550123",
        from_number="+15555550100",
        sip_trunk_id="ST_example",
        pronunciations={"Northwind": "North wind", "INV-": "invoice "},
        mock=mock,
    )
