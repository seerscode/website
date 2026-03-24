import { j as json } from './index-B_P6GQpZ.js';

async function POST({ request }) {
  try {
    const { messages } = await request.json();
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return json({ error: "Invalid messages format" }, { status: 400 });
    }
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return json({
        error: "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
      }, { status: 500 });
    }
    const openaiMessages = [
      {
        role: "system",
        content: "You are a helpful AI assistant. Be concise, accurate, and helpful in your responses."
      },
      ...messages
    ];
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "ft:gpt-4.1-2025-04-14:personal::CAGWwBQp",
        messages: openaiMessages,
        max_tokens: 1e3,
        temperature: 0.7,
        stream: false
      })
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error("OpenAI API error:", errorData);
      if (response.status === 401) {
        return json({
          error: "Invalid OpenAI API key. Please check your API key configuration."
        }, { status: 500 });
      } else if (response.status === 429) {
        return json({
          error: "Rate limit exceeded. Please try again later."
        }, { status: 429 });
      } else if (response.status === 500) {
        return json({
          error: "OpenAI service is temporarily unavailable. Please try again later."
        }, { status: 500 });
      } else {
        return json({
          error: `OpenAI API error: ${errorData.error?.message || "Unknown error"}`
        }, { status: 500 });
      }
    }
    const data = await response.json();
    if (!data.choices || data.choices.length === 0) {
      return json({ error: "No response from OpenAI" }, { status: 500 });
    }
    const message = data.choices[0].message?.content;
    if (!message) {
      return json({ error: "Empty response from OpenAI" }, { status: 500 });
    }
    return json({ message });
  } catch (error) {
    console.error("Chat API error:", error);
    if (error.name === "TypeError" && error.message.includes("fetch")) {
      return json({
        error: "Network error. Please check your internet connection and try again."
      }, { status: 500 });
    }
    return json({
      error: "An unexpected error occurred. Please try again."
    }, { status: 500 });
  }
}

export { POST };
//# sourceMappingURL=_server-DwaxhvHS.js.map
