# Keeps $...$ and $$...$$ math in Markdown posts away from kramdown.
#
# kramdown only knows $$...$$ math and starts a table on any line containing
# "|", so Obsidian-style inline math such as $\pi_\theta(\cdot | x)$ gets split
# into table cells, and "_", "*", "\{" inside $...$ are read as Markdown
# (emphasis, escapes). Before rendering, every math span outside code is swapped
# for a plain-text token; after rendering, each token becomes \(...\) (inline)
# or \[...\] (display), which MathJax typesets (see _includes/mathjax.html).
#
# Inline math follows pandoc's rule so prices stay text: the opening $ must be
# followed by a non-space, the closing $ preceded by a non-space and not
# followed by a digit ("$5 and $10" is not math). Write \$ for a literal dollar.
require "cgi"

module Masani
  module ProtectMath
    TOKEN_RE = /MJXPH(\d+)END/
    FENCE_RE = /\A\s*(`{3,}|~{3,})/
    SPAN_RE = /
        (?<code>(?<ticks>`+)(?:(?!\n[ \t]*\n).)+?\k<ticks>)   # inline code: keep
      | (?<escaped>\\\$)                                       # \$: keep
      | \$\$(?<display>.+?)\$\$                                # display math
      | \$(?<inline>[^\s$](?:[^$\n]*?[^\s$\\])?)\$(?![$\d])    # inline math
    /mx

    module_function

    def markdown?(doc)
      %w(.md .markdown .mkd .mkdn).include?(doc.extname.to_s.downcase)
    end

    # Returns [text with tokens, spans]; spans[i] = [:inline|:display, latex].
    def protect(text)
      spans = []
      segments(text).map do |kind, chunk|
        next chunk if kind == :code

        chunk.gsub(SPAN_RE) do
          m = Regexp.last_match
          if m[:display]
            spans << [:display, m[:display]]
          elsif m[:inline]
            spans << [:inline, m[:inline]]
          else
            next m[0]
          end
          "MJXPH#{spans.length - 1}END"
        end
      end.join.then { |out| [out, spans] }
    end

    # Splits Markdown into [:text, str] and [:code, str] pieces (fenced blocks).
    def segments(text)
      pieces = []
      buffer = +""
      fence = nil
      text.each_line do |line|
        if fence.nil? && (m = FENCE_RE.match(line))
          pieces << [:text, buffer] unless buffer.empty?
          buffer = +line
          fence = m[1]
        elsif fence && line.strip.start_with?(fence[0] * fence.length) && line.strip.delete(fence[0]).empty?
          buffer << line
          pieces << [:code, buffer]
          buffer = +""
          fence = nil
        else
          buffer << line
        end
      end
      pieces << [(fence ? :code : :text), buffer] unless buffer.empty?
      pieces
    end

    def restore(html, spans)
      html.gsub(TOKEN_RE) do
        kind, latex = spans[Regexp.last_match(1).to_i]
        next Regexp.last_match(0) unless kind

        escaped = CGI.escapeHTML(latex)
        kind == :display ? "\\[#{escaped}\\]" : "\\(#{escaped}\\)"
      end
    end
  end
end

Jekyll::Hooks.register :posts, :pre_render do |doc|
  next unless Masani::ProtectMath.markdown?(doc)

  doc.content, spans = Masani::ProtectMath.protect(doc.content)
  doc.instance_variable_set(:@protected_math, spans)
end

Jekyll::Hooks.register :posts, :post_render do |doc|
  spans = doc.instance_variable_get(:@protected_math)
  next if spans.nil? || spans.empty?

  doc.content = Masani::ProtectMath.restore(doc.content, spans)
  doc.output = Masani::ProtectMath.restore(doc.output, spans) if doc.output
end
