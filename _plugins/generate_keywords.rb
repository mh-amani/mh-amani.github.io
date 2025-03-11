Jekyll::Hooks.register :site, :post_write do |site|
    all_keywords = []

    site.posts.docs.each do |post|
        keywords = post.data["keywords"]
        next unless keywords

        keywords.split(",").each do |keyword|
            keyword.strip!
            all_keywords << keyword unless all_keywords.include?(keyword)
        end
    end

    keyword_dir = File.join(site.source, "blog/keywords")
    FileUtils.mkdir_p(keyword_dir)

    all_keywords.each do |keyword|
        slug = Jekyll::Utils.slugify(keyword)
        File.write(File.join(keyword_dir, "#{slug}.md"), "---\nlayout: blog\npermalink: /blog/keywords/#{slug}/\nkeyword: #{keyword}\n---")
    end
end
