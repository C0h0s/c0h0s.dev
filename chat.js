import { HfInference } from '@huggingface/inference'
import { HuggingFaceStream, StreamingTextResponse } from 'ai'
import cors from 'cors'

// Initialize the Hugging Face Inference client
const Hf = new HfInference(process.env.HUGGINGFACE_API_KEY)

// Ensure the API key is set
if (!process.env.HUGGINGFACE_API_KEY) {
  throw new Error('HUGGINGFACE_API_KEY is not set in the environment variables')
}

// Initialize CORS middleware
const corsMiddleware = cors({
  origin: '*', // Be cautious with this in production
  methods: ['POST'],
})

export const config = {
  runtime: 'edge',
}

export default async function handler(req) {
  // Handle CORS
  await new Promise((resolve, reject) => {
    corsMiddleware(req, {
      end: (result) => resolve(result),
      setHeader: (key, value) => req.headers.set(key, value),
    }, reject)
  })

  // Ensure the request method is POST
  if (req.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405 })
  }

  try {
    // Extract the messages from the request body
    const { messages } = await req.json()

    // Initialize conversation history
    let conversationHistory = ''

    // Build the conversation history from the messages
    for (const message of messages) {
      conversationHistory += `${message.role}: ${message.content}\n`
    }

    // Generate a response using the Hugging Face Inference API
    const response = await Hf.textGenerationStream({
      model: 'microsoft/DialoGPT-medium',
      inputs: conversationHistory + 'AI:',
      parameters: {
        max_new_tokens: 200,
        temperature: 0.7,
        top_p: 0.95,
        repetition_penalty: 1.2,
      },
    })

    // Create a stream from the response
    const stream = HuggingFaceStream(response)

    // Return a StreamingTextResponse, which can be consumed by the client
    return new StreamingTextResponse(stream)
  } catch (error) {
    console.error('Error in chat API:', error)
    return new Response(JSON.stringify({ error: 'An error occurred while processing your request' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    })
  }
}