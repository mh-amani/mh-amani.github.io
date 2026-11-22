# Builds one in-memory page per blog keyword at /blog/keywords/<slug>/.
#
# Each page uses the `blog` layout with `page.keyword` set, which filters the
# post list. Pages are created during generation (not written to the source
# tree after the build), so every keyword gets a page on a clean CI checkout.
# Unlisted posts are skipped, matching the loops in _layouts/blog.html.
module Masani
  class KeywordPageGenerator < Jekyll::Generator
    safe true
    priority :low # before jekyll-sitemap (:lowest), so keyword pages are in the sitemap

    def generate(site)
      keywords = []
      site.posts.docs.each do |post|
        next if post.data["unlisted"]

        post.data["keywords"].to_s.split(",").each do |raw|
          keyword = raw.strip
          keywords << keyword unless keyword.empty? || keywords.include?(keyword)
        end
      end

      keywords.each do |keyword|
        slug = Jekyll::Utils.slugify(keyword)
        page = Jekyll::PageWithoutAFile.new(site, site.source, "blog/keywords/#{slug}", "index.html")
        page.content = ""
        page.data["layout"] = "blog"
        page.data["permalink"] = "/blog/keywords/#{slug}/"
        page.data["keyword"] = keyword
        site.pages << page
      end
    end
  end
end
