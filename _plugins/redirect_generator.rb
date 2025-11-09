module Jekyll
  class RedirectGenerator < Generator
    safe true
    priority :low

    def generate(site)
      # Tạo redirect pages cho các URL cũ
      create_chapter_redirects(site)
      create_post_redirects(site)
    end

    private

    def create_chapter_redirects(site)
      # Redirect từ /contents/chapter01/ đến /contents/en/chapter01/
      (0..25).each do |i|
        chapter_num = i.to_s.rjust(2, '0')
        old_path = "contents/chapter#{chapter_num}/index.html"
        new_path = "/contents/en/chapter#{chapter_num}/"
        
        # Kiểm tra xem chapter có tồn tại không
        en_chapter_path = File.join(site.source, 'contents', 'en', "chapter#{chapter_num}")
        if Dir.exist?(en_chapter_path)
          create_redirect_page(site, old_path, new_path)
        end
      end
    end

    def create_post_redirects(site)
      # Tạo redirect cho các posts cũ
      site.posts.docs.each do |post|
        if post.data['lang'] == 'en' && post.data['chapter']
          chapter = post.data['chapter']
          
          # Tạo old-style URL
          post_basename = File.basename(post.path, '.md')
          old_url = "/contents/chapter#{chapter}/#{post_basename}/"
          
          # Redirect đến URL mới
          create_redirect_page(site, old_url.sub(/^\//, '') + 'index.html', post.url)
        end
      end
    end

    def create_redirect_page(site, from_path, to_url)
      # Tạo nội dung redirect page
      redirect_content = <<~HTML
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>Redirecting...</title>
          <link rel="canonical" href="#{to_url}">
          <script>location="#{to_url}"</script>
          <meta http-equiv="refresh" content="0; url=#{to_url}">
          <meta name="robots" content="noindex">
        </head>
        <body>
          <h1>Redirecting...</h1>
          <p><a href="#{to_url}">Click here if you are not redirected automatically</a></p>
        </body>
        </html>
      HTML

      # Tạo Jekyll page object
      redirect_page = PageWithoutAFile.new(site, site.source, File.dirname(from_path), File.basename(from_path))
      redirect_page.content = redirect_content
      redirect_page.data['layout'] = nil
      
      site.pages << redirect_page
    end
  end

  class PageWithoutAFile < Page
    def initialize(site, base, dir, name)
      @site = site
      @base = base
      @dir = dir
      @name = name
      
      self.process(name)
      self.data = {}
    end
    
    def read_yaml(*)
      # Do nothing
    end
  end
end
