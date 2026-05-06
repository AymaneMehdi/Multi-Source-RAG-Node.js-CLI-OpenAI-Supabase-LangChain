// ==========================
// STEP 0: IMPORTS
// ==========================

// Load environment variables from .env
import "dotenv/config";

// Node.js file system module
// Used to read local files like docs.txt, faq.txt, api-data.json
import fs from "fs";

// Node.js readline module
// Used to ask questions from the terminal
import readline from "readline";

// Supabase client
// Used to connect Node.js with Supabase database
import { createClient } from "@supabase/supabase-js";

// OpenAI embedding model and chat model from LangChain
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";

// Supabase vector store integration from LangChain
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";

// LangChain Document class
// LangChain works with documents in this format
import { Document } from "@langchain/core/documents";

// LangChain prompt helper
// Used to create a clean prompt for the LLM
import { ChatPromptTemplate } from "@langchain/core/prompts";


// ==========================
// STEP 1: CREATE SUPABASE CLIENT
// ==========================

// This connects our app to Supabase.
// Supabase will store:
// - text chunks
// - metadata
// - embedding vectors
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);


// ==========================
// STEP 2: CREATE EMBEDDING MODEL
// ==========================

// Embeddings convert text into vectors.
// Example:
// "What is RAG?"
// becomes:
// [0.12, -0.44, 0.91, ...]
//
// LangChain uses this model automatically when saving/searching documents.
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
});


// ==========================
// STEP 3: CREATE CHAT MODEL
// ==========================

// This is the LLM that generates the final answer.
// It receives:
// - retrieved context
// - user question
//
// temperature: 0 means more stable and less random answers.
const llm = new ChatOpenAI({
  model: "gpt-4.1-mini",
  temperature: 0,
});


// ==========================
// STEP 4: LOAD docs.txt SOURCE
// ==========================

// This function reads docs.txt.
// Then it splits the file into small chunks.
// Each chunk becomes a LangChain Document.
function loadDocsSource() {
  const text = fs.readFileSync("docs.txt", "utf-8");

  return text
    .split(".")
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .map(
      (chunk) =>
        new Document({
          pageContent: chunk,
          metadata: {
            source: "docs.txt",
            type: "text",
          },
        })
    );
}


// ==========================
// STEP 5: LOAD faq.txt SOURCE
// ==========================

// This function reads faq.txt.
// FAQ content is usually separated by empty lines.
//
// Example:
// Question: What is RAG?
// Answer: RAG means Retrieval-Augmented Generation.
//
// Question: What is embedding?
// Answer: Embedding converts text into vectors.
function loadFAQSource() {
  const text = fs.readFileSync("faq.txt", "utf-8");

  return text
    .split("\n\n")
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .map(
      (chunk) =>
        new Document({
          pageContent: chunk,
          metadata: {
            source: "faq.txt",
            type: "faq",
          },
        })
    );
}


// ==========================
// STEP 6: LOAD api-data.json SOURCE
// ==========================

// This function reads structured JSON data.
//
// Example JSON:
// [
//   {
//     "title": "Vector Database",
//     "content": "A vector database stores embeddings."
//   }
// ]
//
// We convert JSON objects into text because embeddings need text input.
function loadJSONSource() {
  const raw = fs.readFileSync("api-data.json", "utf-8");
  const items = JSON.parse(raw);

  return items.map(
    (item) =>
      new Document({
        pageContent: `${item.title}: ${item.content}`,
        metadata: {
          source: "api-data.json",
          type: "json",
        },
      })
  );
}


// ==========================
// STEP 7: LOAD ALL SOURCES
// ==========================

// This combines all sources into one array.
//
// Final result:
// [
//   Document from docs.txt,
//   Document from faq.txt,
//   Document from api-data.json
// ]
function loadAllSources() {
  return [
    ...loadDocsSource(),
    ...loadFAQSource(),
    ...loadJSONSource(),
  ];
}


// ==========================
// STEP 8: CREATE LANGCHAIN VECTOR STORE
// ==========================

// Vector store = place where vectors are stored and searched.
//
// Here we connect LangChain to Supabase pgvector.
// LangChain will use:
// - embeddings model to create vectors
// - Supabase table to store/search documents
// - match_documents RPC function to find similar chunks
const vectorStore = new SupabaseVectorStore(embeddings, {
  client: supabase,
  tableName: "documents",
  queryName: "match_documents",
});


// ==========================
// STEP 9: INSERT SOURCES INTO SUPABASE
// ==========================

// This is the ingestion step.
//
// Ingestion means:
// 1. Load all source files
// 2. Convert them into LangChain Documents
// 3. Create embeddings for each document
// 4. Store content + metadata + vectors in Supabase
//
// Run this only one time.
// After inserting, comment it again to avoid duplicate data.
async function insertSources() {
  const documents = loadAllSources();

  await vectorStore.addDocuments(documents);

  console.log("All sources inserted into Supabase.");
}


// ==========================
// STEP 10: ASK USER QUESTION FROM TERMINAL
// ==========================

// This lets the user type a question in the terminal.
function askQuestion() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return new Promise((resolve) => {
    rl.question("Ask a question: ", (question) => {
      rl.close();
      resolve(question);
    });
  });
}


// ==========================
// STEP 11: GENERATE ANSWER USING RAG
// ==========================

// This function receives:
// - the user question
// - the retrieved documents from Supabase
//
// Then it builds the context and sends it to the LLM.
//
// Important:
// The LLM does NOT search the database.
// Supabase searches first.
// Then the LLM only answers from the retrieved context.
async function generateAnswer(question, docs) {
  const context = docs
    .map((doc) => {
      return `[Source: ${doc.metadata.source} | Type: ${doc.metadata.type}]
${doc.pageContent}`;
    })
    .join("\n\n");

  // This prompt tells the model how to answer.
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer ONLY using the provided context. Mention the source if useful. If the answer is not in the context, say you don't know.",
    ],
    [
      "user",
      `Context:
{context}

Question:
{question}`,
    ],
  ]);

  // This creates a simple LangChain chain:
  // prompt -> llm
  //
  // Meaning:
  // 1. Fill the prompt with context and question
  // 2. Send it to the LLM
  // 3. Return the model response
  const chain = prompt.pipe(llm);

  const response = await chain.invoke({
    context,
    question,
  });

  return response.content;
}


// ==========================
// STEP 12: MAIN APP FLOW
// ==========================

// Main RAG flow:
//
// 1. Optional ingestion
// 2. Ask user question
// 3. Convert question into embedding
// 4. Search similar documents in Supabase
// 5. Build context from results
// 6. Send context + question to LLM
// 7. Print final answer
async function main() {
  // FIRST RUN ONLY:
  //
  // Uncomment this line one time:
  // await insertSources();
  //
  // Then run:
  // node index.js
  //
  // After data is inserted, comment it again.
  // await insertSources();

  const question = await askQuestion();

  // similaritySearch does this automatically:
  // 1. Converts the question into an embedding
  // 2. Sends it to Supabase match_documents
  // 3. Returns the most similar documents
  const results = await vectorStore.similaritySearch(question, 5);

  console.log("\nRetrieved Sources:");

  results.forEach((doc) => {
    console.log(`- ${doc.metadata.source} (${doc.metadata.type})`);
  });

  const answer = await generateAnswer(question, results);

  console.log("\nAnswer:");
  console.log(answer);
}


// ==========================
// STEP 13: RUN APP
// ==========================

main();