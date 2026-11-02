"""Test text. Caller utterances feed the STT corpus; agent replies feed the TTS benchmark."""

CALLER_UTTERANCES: dict[str, list[str]] = {
    "en": [
        "Hi, I'm calling about an invoice that was charged twice this month.",
        "Yes.",
        "No, that's not right.",
        "Can you transfer me to a person please?",
        "My purchase order number is four five seven two nine one.",
        "I need to reset my password for the supplier portal.",
        "The delivery was supposed to arrive last Tuesday but it never showed up.",
        "Sure, go ahead.",
        "Could you repeat that one more time?",
        "We moved our warehouse to Rotterdam and the shipping address is still the old one.",
        "I'd like to cancel the subscription and get a refund for the remaining months.",
        "That's correct.",
        "Hold on a second, let me grab my account details.",
        "The approval workflow is stuck at the second step for three days now.",
        "Okay, thank you so much, that's all I needed.",
        "My name is Siobhan Okonkwo-Fitzgerald.",
        "Is there any way to expedite the replacement part?",
        "We're getting an error when we try to post the goods receipt.",
        "I already tried that twice and it didn't work.",
        "Wait, sorry, can you go back?",
    ],
    "es": [
        "Hola, llamo porque me cobraron dos veces la misma factura.",
        "Sí, es correcto.",
        "Necesito cambiar la dirección de entrega de mi pedido.",
        "¿Me puede comunicar con un agente, por favor?",
        "El paquete no ha llegado y ya pasó una semana.",
        "Quiero cancelar mi suscripción a partir del próximo mes.",
    ],
    "fr": [
        "Bonjour, j'appelle au sujet d'une facture qui a été débitée deux fois.",
        "Oui, c'est exact.",
        "Je voudrais modifier l'adresse de livraison de ma commande.",
        "Pouvez-vous me passer un conseiller, s'il vous plaît?",
        "Le colis n'est toujours pas arrivé depuis une semaine.",
        "Je souhaite résilier mon abonnement à la fin du mois.",
    ],
    "de": [
        "Guten Tag, mir wurde dieselbe Rechnung zweimal abgebucht.",
        "Ja, das stimmt.",
        "Ich möchte die Lieferadresse meiner Bestellung ändern.",
        "Können Sie mich bitte mit einem Mitarbeiter verbinden?",
        "Das Paket ist seit einer Woche nicht angekommen.",
        "Ich möchte mein Abonnement zum Monatsende kündigen.",
    ],
    "pt": [
        "Olá, estou ligando porque cobraram a mesma fatura duas vezes.",
        "Sim, está certo.",
        "Preciso mudar o endereço de entrega do meu pedido.",
        "Pode me transferir para um atendente, por favor?",
        "A encomenda ainda não chegou e já faz uma semana.",
        "Quero cancelar minha assinatura no fim do mês.",
    ],
    "hi": [
        "नमस्ते, मेरे बिल का भुगतान दो बार कट गया है।",
        "हाँ, यह सही है।",
        "मुझे अपने ऑर्डर का डिलीवरी पता बदलना है।",
        "क्या आप मुझे किसी एजेंट से बात करवा सकते हैं?",
        "मेरा पार्सल एक हफ्ते से नहीं आया है।",
        "मैं अगले महीने से अपनी सदस्यता रद्द करना चाहता हूँ।",
    ],
}

# (language, Inworld voice) pairs used to speak the caller corpus. English uses several accents.
CORPUS_VOICES: list[tuple[str, str, str]] = [
    # (corpus language tag, TTS language, voice)
    ("en", "en", "Ashley"),
    ("en", "en", "Edward"),
    ("en-GB", "en", "Olivia"),
    ("es", "es", "Diego"),
    ("fr", "fr", "Ashley"),
    ("de", "de", "Edward"),
    ("pt", "pt", "Ashley"),
    ("hi", "hi", "Edward"),
]

AGENT_REPLIES: dict[str, list[str]] = {
    "en": [
        "Sure, I can help with that.",
        "I see two charges for invoice nine eight one two on October third. I've flagged the duplicate, and you'll get a refund within five business days. Is there anything else I can do?",
        "Your purchase order 457291 was approved on March 14th for $12,480.50.",
        "Please send the documents to accounts.payable@example-corp.com, and include your vendor ID, V as in Victor, 2 0 7 7.",
        "I'm sorry about the delay. The shipment left our Rotterdam warehouse yesterday. It should arrive by Friday. You'll get a tracking link by text message.",
        "Got it.",
        "To reset your password, open the supplier portal, click forgot password, and follow the link we email you. The link expires after thirty minutes. If it doesn't arrive, check your spam folder. Would you like me to send it now?",
        "Your appointment is confirmed for Tuesday, November 4th at 2:30 PM Central Time.",
        "The approval is waiting on your finance manager, Siobhan Okonkwo-Fitzgerald. I've sent her a reminder.",
        "Let me transfer you to a specialist now. Please stay on the line.",
    ],
    "es": [
        "Claro, con gusto le ayudo.",
        "Veo dos cargos por la misma factura. Ya marqué el duplicado y recibirá el reembolso en cinco días hábiles. ¿Hay algo más en lo que pueda ayudarle?",
        "Su pedido llegará el viernes. Le enviaremos el enlace de seguimiento por mensaje de texto.",
    ],
    "fr": [
        "Bien sûr, je peux vous aider.",
        "Je vois deux prélèvements pour la même facture. J'ai signalé le doublon et vous serez remboursé sous cinq jours ouvrés. Puis-je faire autre chose pour vous?",
        "Votre commande arrivera vendredi. Vous recevrez le lien de suivi par SMS.",
    ],
    "de": [
        "Natürlich, ich helfe Ihnen gerne.",
        "Ich sehe zwei Abbuchungen für dieselbe Rechnung. Ich habe die doppelte markiert, und Sie erhalten die Erstattung innerhalb von fünf Werktagen. Kann ich sonst noch etwas für Sie tun?",
        "Ihre Bestellung kommt am Freitag an. Den Sendungslink schicken wir Ihnen per SMS.",
    ],
    "pt": [
        "Claro, posso ajudar com isso.",
        "Vejo duas cobranças da mesma fatura. Já marquei a duplicada e o reembolso sai em cinco dias úteis. Posso ajudar em mais alguma coisa?",
    ],
    "hi": [
        "ज़रूर, मैं आपकी मदद कर सकता हूँ।",
        "मुझे एक ही बिल के दो भुगतान दिख रहे हैं। मैंने दोहरे भुगतान को चिह्नित कर दिया है और पाँच कार्यदिवसों में आपको रिफंड मिल जाएगा।",
    ],
}
