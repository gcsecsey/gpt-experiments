import fetch from 'node-fetch'
import * as cheerio from 'cheerio'
import { HNSWLib } from 'langchain/vectorstores'
import { OpenAIEmbeddings } from 'langchain/embeddings'
import { CharacterTextSplitter } from 'langchain/text_splitter'
import * as dotenv from 'dotenv' // see https://github.com/motdotla/dotenv#how-do-i-use-dotenv-with-import
import Turndown from 'turndown'

// Load .env.local into process.env
dotenv.config({ path: '.env.local' })

const urls = [
  'https://woocommerce.com/products/woocommerce-payments/',
  'https://woocommerce.com/products/woocommerce-google-analytics/',
  'https://woocommerce.com/products/tax/',
  'https://woocommerce.com/products/woocommerce-paypal-payments/',
  'https://woocommerce.com/products/stripe/',
]

const uniqueUrls = urls.filter((url, index) => urls.indexOf(url) === index)

async function run() {
  const posts = []
  const metadatas = []
  const turndownService = new Turndown({
    hr: '',
    codeBlockStyle: 'fenced',
    headingStyle: 'atx',
  })

  for (const url of uniqueUrls) {
    const response = await fetch(url)
    const html = await response.text()
    const $ = cheerio.load(html)
    const postContentStr = $('main.wccom-product-single__main')
      .text()
      .replace(/(\r\n|\n|\r|\t)/gm, ' ')

    let postContent = $('main.wccom-product-single__main')
    // Remove "Back to top" links from HTML
    postContent.find('a').each((i, el) => {
      if ($(el).text().includes('Back to top')) {
        $(el).remove()
      }
    })
    const postContentHTML = postContent.html() || ''

    let markdown = turndownService
      .turndown(postContentHTML)
      // remove newlines
      .replace(/\n/g, ' ')
      // remove multiple spaces
      .replace(/ +(?= )/g, '')

    console.log(postContentStr.length, markdown.length)

    posts.push(markdown)
    metadatas.push({ url })

    console.log(`${url} fetched`)
  }

  /* Split the text into chunks */
  console.log('Splitting text into chunks')
  const textSplitter = new CharacterTextSplitter({
    chunkSize: 1000,
    separator: '. ',
  })

  const docs = await textSplitter.createDocuments(posts, metadatas)

  /* Create the vectorstore */
  console.log('Creating vectorstore')
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings())
  vectorStore.save('public/vectorstores/wcpay-docs')

  console.log(`Vectorstore created`)
}

run()
