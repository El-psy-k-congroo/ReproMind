import MarkdownIt from 'markdown-it'
import DOMPurify from 'dompurify'
import katex from 'katex'

const markdown = new MarkdownIt({ html: false, linkify: true, breaks: true })
markdown.validateLink = (url: string) => /^(https?:|mailto:)/i.test(url)
const defaultLinkOpen = markdown.renderer.rules.link_open
markdown.renderer.rules.link_open = (tokens, index, options, env, self) => {
  tokens[index].attrSet('target', '_blank')
  tokens[index].attrSet('rel', 'noopener noreferrer')
  return defaultLinkOpen ? defaultLinkOpen(tokens, index, options, env, self) : self.renderToken(tokens, index, options)
}

export function renderResearchText(source: string): string {
  const math: string[] = []
  const withMath = source
    .replace(/\$\$([\s\S]+?)\$\$/g, (_whole, expression: string) => {
      const token = `REPROMINDMATH${math.length}END`
      math.push(katex.renderToString(expression, { displayMode: true, throwOnError: false, trust: false }))
      return token
    })
    .replace(/(?<!\$)\$([^\n$]+?)\$(?!\$)/g, (_whole, expression: string) => {
      const token = `REPROMINDMATH${math.length}END`
      math.push(katex.renderToString(expression, { displayMode: false, throwOnError: false, trust: false }))
      return token
    })
  const rendered = markdown.render(withMath).replace(/REPROMINDMATH(\d+)END/g, (_whole, index: string) => math[Number(index)] || '')
  return DOMPurify.sanitize(rendered)
}
