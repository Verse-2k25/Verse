"use client"
import { useState, useRef, useEffect } from "react"
import { Send } from "lucide-react"

const Chatbot = () => {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(scrollToBottom, [messages])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = { role: "user", content: input }
    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: [...messages, userMessage],
        }),
      })

      const data = await response.json()

      if (response.ok) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.message,
          },
        ])
      } else {
        throw new Error(data.error || "Failed to get response")
      }
    } catch (error) {
      console.error("Error:", error)
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, there was an error processing your request.",
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#EDE5D9] p-4 flex items-center justify-center">
      <div className="w-full max-w-4xl bg-white rounded-2xl shadow-lg overflow-hidden">
        <div className="bg-[#41362A] p-6 text-[#EDE5D9]">
          <h1 className="text-2xl font-bold text-center ">Meewli ♪</h1>
        </div>
        <div className="p-6">
          <div className="space-y-4 mb-6 h-[500px] overflow-y-auto pr-4">
            {messages.map((message, index) => (
              <div key={index} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[80%] rounded-2xl p-4 ${
                    message.role === "user" ? "bg-[#D77E1D] text-white" : "bg-[#EDE5D9] text-[#41362A]"
                  }`}
                >
                  {message.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-[#EDE5D9] text-[#41362A] rounded-2xl p-4">Thinking...</div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSubmit} className="flex gap-4">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 p-4 border border-[#41362A] rounded-full focus:outline-none focus:ring-2 focus:ring-[#D77E1D] bg-[#EDE5D9] text-[#41362A] placeholder-[#41362A]/50"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading}
              className="bg-[#D77E1D] text-white p-4 rounded-full hover:bg-[#D77E1D]/80 disabled:bg-[#D77E1D]/50 transition-colors duration-200"
            >
              <Send className="w-6 h-6" />
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default Chatbot

