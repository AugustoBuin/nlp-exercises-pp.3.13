import re
import spacy
import random
from flask import Flask, request, render_template_string
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
nlp = spacy.load("pt_core_news_md")

# Perguntas frequentes e respostas
faq = {
    "como instalo um programa": "Para instalar um programa, você pode utilizar o gerenciador de pacotes do seu sistema. Por exemplo: sudo apt install nome-do-programa",
    "como remover um programa": "Você pode remover programas com: sudo apt remove nome-do-programa",
    "como atualizo o sistema": "Use: sudo apt update && sudo apt upgrade",
    "como vejo programas instalados": "Você pode usar: dpkg --list ou flatpak list",
    "o que você pode fazer": "Posso responder perguntas técnicas relacionadas a gerenciamento de pacotes e operações básicas no sistema.",
}


# Função de normalização
def normalize(text):
    return " ".join(
        [
            token.lemma_.lower()
            for token in nlp(text)
            if not token.is_punct and not token.is_space
        ]
    )


# Preparar dados
questions = list(faq.keys())
answers = list(faq.values())
normalized_questions = [normalize(q) for q in questions]
question_vectors = [nlp(q).vector for q in normalized_questions]

# Mensagens alternativas de fallback
fallback_responses = [
    "Desculpe, não entendi. Pode tentar de outro jeito?",
    "Hmm, essa é nova pra mim. Reformule por favor.",
    "Ainda não sei como responder isso. Pode perguntar de outra forma?",
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt">
<head>
    <meta charset="UTF-8">
    <title>ChatBot Semântico</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2rem; }
        .chat-box { margin-top: 2rem; }
        .msg { margin: 0.5rem 0; }
        .user { color: #1e88e5; }
        .bot { color: #43a047; }
    </style>
</head>
<body>
    <h2>ChatBot - Ajuda Técnica</h2>
    <form method="POST">
        <input type="text" name="user_input" placeholder="Digite sua pergunta aqui" size="50" required autofocus>
        <button type="submit">Enviar</button>
    </form>

    <div class="chat-box">
        {% for u, r in history %}
            <p class="msg user"><strong>Você:</strong> {{ u }}</p>
            <p class="msg bot"><strong>ChatBot:</strong> {{ r }}</p>
        {% endfor %}
    </div>
</body>
</html>
"""

# Armazenar histórico
chat_history = []


@app.route("/", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        user_input = request.form["user_input"].strip()

        if user_input.lower() in ["sair", "quero sair"]:
            bot_response = "Até mais! Qualquer coisa, estou por aqui."
        else:
            norm_input = normalize(user_input)
            user_vec = nlp(norm_input).vector
            sims = cosine_similarity([user_vec], question_vectors)
            max_sim_idx = sims.argmax()

            if sims[0, max_sim_idx] > 0.60:
                bot_response = answers[max_sim_idx]
            else:
                bot_response = random.choice(fallback_responses)

        chat_history.append((user_input, bot_response))

    return render_template_string(HTML_TEMPLATE, history=chat_history)


if __name__ == "__main__":
    app.run(debug=True)
