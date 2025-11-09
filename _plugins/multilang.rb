module Jekyll
  class LanguageTag < Liquid::Tag
    def initialize(tag_name, lang, tokens)
      super
      @lang = lang.strip
    end

    def render(context)
      site = context.registers[:site]
      page = context.registers[:page]
      
      current_lang = page['lang'] || site.config['default_lang']
      
      if current_lang == @lang
        return "active"
      else
        return ""
      end
    end
  end

  class TranslateTag < Liquid::Tag
    def initialize(tag_name, key, tokens)
      super
      @key = key.strip
    end

    def render(context)
      site = context.registers[:site]
      page = context.registers[:page]
      
      current_lang = page['lang'] || site.config['default_lang']
      
      if site.config['t'] && site.config['t'][current_lang] && site.config['t'][current_lang][@key]
        return site.config['t'][current_lang][@key]
      else
        return @key
      end
    end
  end

  class LanguageSwitchTag < Liquid::Tag
    def render(context)
      site = context.registers[:site]
      page = context.registers[:page]
      
      current_lang = page['lang'] || site.config['default_lang']
      other_lang = current_lang == 'en' ? 'vi' : 'en'
      
      # Tạo URL cho ngôn ngữ khác
      current_url = page['url']
      other_url = nil
      
      if current_url.start_with?("/contents/#{current_lang}/")
        potential_url = current_url.gsub("/contents/#{current_lang}/", "/contents/#{other_lang}/")
        
        # Kiểm tra xem trang tương ứng có tồn tại không (tìm trong cả pages và posts)
        target_page = site.pages.find { |p| p.url == potential_url && p.data['lang'] == other_lang }
        target_post = site.posts.docs.find { |p| p.url == potential_url && p.data['lang'] == other_lang }
        
        if target_page || target_post
          other_url = potential_url
        else
          # Nếu không tồn tại, tìm bài viết tương ứng bằng cách so sánh chapter và order
          # Tìm bài viết hiện tại để lấy chapter và order
          current_post = site.posts.docs.find { |p| p.url == current_url }
          
          if current_post && current_post.data['chapter'] && current_post.data['order']
            # Tìm bài viết có cùng chapter và order trong ngôn ngữ khác
            matching_post = site.posts.docs.find do |post|
              post.data['lang'] == other_lang && 
              post.data['chapter'] == current_post.data['chapter'] &&
              post.data['order'] == current_post.data['order']
            end
            
            if matching_post
              other_url = matching_post.url
            else
              # Fallback về trang chapter tương ứng
              chapter_num = current_post.data['chapter']
              fallback_page = site.pages.find { |p| p.url == "/contents/#{other_lang}/chapter#{chapter_num}/" && p.data['lang'] == other_lang }
              other_url = fallback_page ? "/contents/#{other_lang}/chapter#{chapter_num}/" : "/contents/#{other_lang}/chapter00/"
            end
          else
            # Fallback về trang chapter00
            fallback_page = site.pages.find { |p| p.url == "/contents/#{other_lang}/chapter00/" && p.data['lang'] == other_lang }
            other_url = fallback_page ? "/contents/#{other_lang}/chapter00/" : "/"
          end
        end
      elsif current_url.start_with?("/contents/")
        # Nếu URL hiện tại không có ngôn ngữ, thêm ngôn ngữ khác
        other_url = current_url.gsub("/contents/", "/contents/#{other_lang}/")
      else
        # Mặc định chuyển về chapter00 của ngôn ngữ khác
        other_url = "/contents/#{other_lang}/chapter00/"
      end
      
      switch_text = site.config['t'][current_lang]['switch_language']
      baseurl = site.config['baseurl'] || ''
      
      return "<a href=\"#{baseurl}#{other_url}\" class=\"language-switch\" title=\"#{switch_text}\">#{switch_text}</a>"
    end
  end
end

Liquid::Template.register_tag('lang', Jekyll::LanguageTag)
Liquid::Template.register_tag('t', Jekyll::TranslateTag)
Liquid::Template.register_tag('language_switch', Jekyll::LanguageSwitchTag)
