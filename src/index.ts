import * as dotenv from "dotenv";
import { OpenAI } from "langchain/llms/openai";
import { PDFLoader } from "langchain/document_loaders/fs/pdf"; // document loader
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"; // text splitter
import { MemoryVectorStore } from "langchain/vectorstores/memory"; // vector db
import { OpenAIEmbeddings } from "langchain/embeddings/openai"; // embeddings
import { RetrievalQAChain } from "langchain/chains";

// config secrets
dotenv.config();

// instantiate a model
const model = new OpenAI({
  modelName: "gpt-3.5-turbo",
  openAIApiKey: process.env.OPENAI_API_KEY,
});


// load a document
const loader = new PDFLoader("data/Micello WorkingSet Data Framework.pdf");
const docs = await loader.load();

// split the document
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 4000,
  chunkOverlap: 200,
});
const splitDocs = await splitter.splitDocuments(docs);

// Load the docs into the vector store
const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY
  }),{});

// Search for the most similar document
// const result = await vectorStore.similaritySearch("rain", 10);

const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
const res = await chain.call({
  query: "What is the Micello WorkingSet data framework?",
});
console.log(res.text);





