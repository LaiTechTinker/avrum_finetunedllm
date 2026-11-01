const API_URL = "http://127.0.0.1:8000";

const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const sendButton = document.getElementById("send-btn");
const chat = document.getElementById("chat");
const status = document.getElementById("status");

function addMessage(role, text = "") {
  const message = document.createElement("div");
  message.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  message.appendChild(bubble);
  chat.appendChild(message);

  chat.scrollTop = chat.scrollHeight;

  return bubble;
}

function setStatus(text, type = "") {
  status.textContent = text;
  status.className = `status ${type}`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();

  if (!question) return;

  // Remove the welcome message after the first question.
  const welcome = document.querySelector(".welcome");
  if (welcome) welcome.remove();

  addMessage("user", question);

  questionInput.value = "";
  questionInput.style.height = "auto";

  sendButton.disabled = true;
  questionInput.disabled = true;
  setStatus("Generating...", "loading");

  const assistantBubble = addMessage("assistant");

  try {
    const response = await fetch(`${API_URL}/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question: question
      })
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    if (!response.body) {
      throw new Error("Streaming is not supported by this response.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { value, done } = await reader.read();

      if (done) break;

      assistantBubble.textContent += decoder.decode(value, {
        stream: true
      });

      chat.scrollTop = chat.scrollHeight;
    }

    // Flush any remaining decoder content.
    assistantBubble.textContent += decoder.decode();

    setStatus("Ready");
  } catch (error) {
    console.error(error);

    assistantBubble.textContent =
      "Sorry, I couldn't connect to the backend. Make sure FastAPI is running and /chat/stream is available.";

    setStatus("Connection error", "error");
  } finally {
    sendButton.disabled = false;
    questionInput.disabled = false;
    questionInput.focus();
  }
});

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 130)}px`;
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
