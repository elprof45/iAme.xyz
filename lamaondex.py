import chromadb
import streamlit as st
from chromadb.utils import embedding_functions

from assistant import get_research_assistant, topics_listes

# Définir la fonction d'embedding par défaut
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Initialiser le client ChromaDB avec une base de données persistante
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Récupérer ou créer une collection de documents dans la base de connaissances
knowledge_base_collection = chroma_client.get_or_create_collection(
    "base1", embedding_function=embedding_function
)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Assistant de Recherche",
    page_icon=":orange_heart:",
    initial_sidebar_state="auto",
    layout="centered",
)
st.markdown("## Base de Connaissances alimentée par SolidaryWord")


# Sélectionner le modèle de LLM via la barre latérale
llm_model = st.sidebar.selectbox(
    "Sélectionner un Modèle", options=["mixtral-8x7b-32768", "llama3-8b-8192"]
)

# Vérifier si le modèle sélectionné est déjà dans l'état de session, sinon l'ajouter
if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = llm_model

# Redémarrer l'assistant si le modèle change
elif st.session_state["llm_model"] != llm_model:
    st.session_state["llm_model"] = llm_model
    st.rerun()


# Fonction pour récupérer les documents pertinents de la base de connaissances
def retrieve_relevant_documents(prompt):
    query_result = knowledge_base_collection.query(
        query_texts=[prompt], n_results=3, include=["documents"]
    )
    context = ""
    for doc in query_result["documents"][0]:
        context += doc
    return context


# Fonction pour générer une réponse basée sur la requête de l'utilisateur
def generate_response(prompt):
    context = retrieve_relevant_documents(prompt)
    research_assistant = get_research_assistant(model=llm_model, context=context)

    # Ajouter la requête de l'utilisateur dans les messages de session
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Afficher la requête dans l'interface utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)

    # Générer et afficher la réponse de l'assistant
    with st.chat_message("assistant"):
        with st.spinner("Génération du rapport..."):
            final_report = ""
            final_report_container = st.empty()

            for delta in research_assistant.run(prompt):
                final_report += delta
                final_report_container.markdown(final_report)

        # Ajouter la réponse de l'assistant dans les messages de session
        st.session_state.messages.append({"role": "assistant", "content": final_report})


# Fonction pour afficher la barre latérale avec les centres d'intérêt
def display_sidebar():
    st.sidebar.markdown("## Vos Centres d'Intérêt")

    # Afficher les boutons pour chaque sujet d'intérêt
    for item in topics_listes:
        if st.sidebar.button(item["title"], key=item["title"]):
            st.session_state["topic"] = item["topic"]
            generate_response(item["topic"])


# Fonction principale de l'application
def app_main_loop():
    # Initialiser les messages dans l'état de session si nécessaire
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Afficher les messages existants (utilisateur et assistant)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Afficher la barre latérale
    display_sidebar()

    # Champ d'entrée pour que l'utilisateur pose des questions
    if prompt := st.chat_input("Entrez votre question ici"):
        generate_response(prompt)

    # Bouton pour redémarrer l'application
    st.sidebar.markdown("---")
    if st.sidebar.button("Redémarrer"):
        st.rerun()


# Exécution de l'application
app_main_loop()
