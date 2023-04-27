import { PDFLoader } from "langchain/document_loaders";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const loader = new PDFLoader("data/stormDataDef.pdf");

const docs = await loader.load();
console.log(docs !== null ? "loaded" : "Not loaded");


const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 4000,
  chunkOverlap: 200,
});

const output = await splitter.splitDocuments(docs);


console.log(output !== null ? "split" : "Not split");
