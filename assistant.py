import os

from dotenv import load_dotenv
from phi.assistant import Assistant
from phi.llm.groq import Groq

# Charger les variables d'environnement depuis un fichier .env
load_dotenv()

# Récupérer l'API key Groq depuis l'environnement
os.getenv("GROQ_API_KEY")


def get_research_assistant(
    context,
    model: str = "llama3-8b-8192",
    debug_mode: bool = True,
) -> Assistant:
    return Assistant(
        name="groq_research_assistant",
        llm=Groq(model="mixtral-8x7b-32768"),
        description=f"""Vous êtes un assistant intelligent spécialisé dans la réponse à des questions. 
        Votre rôle est de fournir des réponses courtes (max 300 mots), précises et concises en vous basant sur le contexte fourni.
        ## Contexte : {context}""",
        instructions=[
            "Lorsque l'utilisateur pose une question, commencez par vérifier si la réponse se trouve dans le contexte :",
            "Si oui, répondez en vous basant sur le contexte.",
            "Si non, répondez en fonction de vos propres connaissances.",
            "Répondez de manière aussi concise que possible et uniquement sur la question, sans commentaire supplémentaire.",
            "If you need to reference the chat history, use the `get_chat_history` tool.",
            "If the users question is unclear, ask clarifying questions to get more information.",
            "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
        ],
        # Show tool calls in the chat
        markdown=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
        # This setting gives the LLM a tool to get chat history
        # read_chat_history=True,
        # This setting adds 6 previous messages from chat history to the messages sent to the LLM
        # num_history_messages=4,
        # Adds chat history to messages
        # add_chat_history_to_messages=True,
    )


topics_listes = [
    {
        "title": "Programmation orientée objet",
        "topic": "Parle-moi de la programmation orientée objet.",
    },
    {
        "title": "Technologie quantique",
        "topic": "Quels sont les derniers développements dans la technologie quantique ?",
    },
    {
        "title": "Éthique de l'intelligence artificielle",
        "topic": "Quelles sont les principales questions éthiques soulevées par l'intelligence artificielle ?",
    },
    {
        "title": "Exploration spatiale",
        "topic": "Quels sont les récents progrès dans l'exploration spatiale ?",
    },
    {
        "title": "Blockchain et cryptomonnaies",
        "topic": "Parle-moi de la blockchain et des cryptomonnaies.",
    },
    {
        "title": "Développement durable",
        "topic": "Quels sont les principaux défis du développement durable aujourd'hui ?",
    },
]
