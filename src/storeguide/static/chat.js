const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const sendButton = document.querySelector("#send-button");
const conversation = document.querySelector("#conversation");

function addMessage({ role, text, sources = [], refused = false, error = false }) {
  const article = document.createElement("article");
  article.className = `message ${role === "user" ? "user-message" : "assistant-message"}${refused ? " refusal" : ""}`;

  if (role !== "user") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "R";
    article.append(avatar);
  }

  const body = document.createElement("div");
  body.className = "message-body";
  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "RetailRAG";
  body.append(label);

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  if (error) paragraph.classList.add("error-message");
  bubble.append(paragraph);
  body.append(bubble);

  if (sources.length > 0) {
    const sourceList = document.createElement("div");
    sourceList.className = "source-list";
    for (const source of sources) {
      const chip = document.createElement("span");
      chip.className = "source-chip";
      chip.textContent = source.title;
      chip.title = `Source: ${source.article_id}`;
      sourceList.append(chip);
    }
    body.append(sourceList);
  }

  article.append(body);
  conversation.append(article);
  conversation.scrollTop = conversation.scrollHeight;
  return article;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  addMessage({ role: "user", text: message });
  input.value = "";
  input.disabled = true;
  sendButton.disabled = true;
  const pending = addMessage({ role: "assistant", text: "Looking through the store policies..." });

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const payload = await response.json();
    pending.remove();
    if (!response.ok) {
      throw new Error(payload.detail || "The support service could not answer right now.");
    }
    addMessage({
      role: "assistant",
      text: payload.answer,
      sources: payload.sources,
      refused: payload.refused,
    });
  } catch (error) {
    pending.remove();
    addMessage({
      role: "assistant",
      text: error instanceof Error ? error.message : "Something went wrong. Please try again.",
      error: true,
    });
  } finally {
    input.disabled = false;
    sendButton.disabled = false;
    input.focus();
  }
});
