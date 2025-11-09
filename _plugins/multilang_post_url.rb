module Jekyll
  class MultiLangPostUrl < Liquid::Tag
    def initialize(tag_name, markup, tokens)
      super
      @markup = markup.strip
    end

    def render(context)
      site = context.registers[:site]
      page = context.registers[:page]
      
      current_lang = page['lang'] || site.config['default_lang']
      
      # Parse the old format: contents/chapter06/21-03-20-06_03_02_convex_function_quadratic_upper_bound
      if @markup.match(/^contents\/chapter(\d+)\/(.+)$/)
        chapter = $1
        post_name = $2
        
        # Try to find the post in the current language first
        target_post = nil
        
        # Look for post in current language
        site.posts.docs.each do |post|
          if post.data['lang'] == current_lang && 
             post.data['chapter'] == chapter && 
             post.basename_without_ext.include?(post_name)
            target_post = post
            break
          end
        end
        
        # Fallback to default language if not found
        if target_post.nil?
          site.posts.docs.each do |post|
            if post.data['chapter'] == chapter && 
               post.basename_without_ext.include?(post_name)
              target_post = post
              break
            end
          end
        end
        
        # Fallback to old format if still not found
        if target_post.nil?
          site.posts.docs.each do |post|
            if post.basename_without_ext.include?(post_name)
              target_post = post
              break
            end
          end
        end
        
        return target_post ? target_post.url : "#post-not-found"
      else
        # Handle regular post_url format
        return Jekyll::Tags::PostUrl.new('post_url', @markup, []).render(context)
      end
    end
  end
end

Liquid::Template.register_tag('multilang_post_url', Jekyll::MultiLangPostUrl)
