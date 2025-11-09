# Plugin để tạo search index cho Lunr.js
# Tạo file JSON chứa tất cả nội dung có thể tìm kiếm

require 'json'
require 'fileutils'

module Jekyll
  class SearchIndexGenerator < Generator
    safe true
    priority :low

    def generate(site)
      # Tạo search index cho từng ngôn ngữ
      %w[en vi].each do |lang|
        search_data = []
        
        # Lấy tất cả posts từ categories chapters
        site.categories.each do |category, posts|
          next unless category.start_with?('chapter')
          
          posts.each do |post|
            # Chỉ lấy posts của ngôn ngữ hiện tại
            post_lang = post.data['lang'] || site.config['default_lang']
            next unless post_lang == lang
            
            # Tạo URL đầy đủ cho post với baseurl
            baseurl = site.config['baseurl'] || ''
            url = post.url
            url = "/#{lang}#{url}" if lang != site.config['default_lang']
            url = "#{baseurl}#{url}"
            
            # Trích xuất nội dung text từ markdown
            content = post.content
            # Loại bỏ HTML tags và markdown syntax
            content = content.gsub(/<[^>]*>/, ' ')
            content = content.gsub(/\$\$.*?\$\$/, ' [MATH] ')  # Thay thế LaTeX math
            content = content.gsub(/\$.*?\$/, ' [MATH] ')      # Thay thế inline math
            content = content.gsub(/[#*`_\[\](){}]/, ' ')      # Loại bỏ markdown syntax
            content = content.gsub(/\s+/, ' ').strip           # Chuẩn hóa whitespace
            
            search_data << {
              id: post.id,
              title: post.data['title'] || '',
              chapter: post.data['chapter'] || '',
              order: post.data['order'] || 0,
              owner: post.data['owner'] || '',
              lesson_type: post.data['lesson_type'] || '',
              content: content,
              url: url,
              lang: lang
            }
          end
        end
        
        # Sắp xếp theo chapter và order
        search_data.sort_by! { |item| [item[:chapter].to_i, item[:order].to_i] }
        
        # Tạo file JSON cho ngôn ngữ này
        filename = lang == site.config['default_lang'] ? 'search-index.json' : "search-index-#{lang}.json"
        
        # Đảm bảo thư mục destination tồn tại
        FileUtils.mkdir_p(site.dest) unless Dir.exist?(site.dest)
        
        begin
          File.open(File.join(site.dest, filename), 'w') do |file|
            file.write(JSON.pretty_generate(search_data))
          end
        rescue => e
          Jekyll.logger.warn "Search Generator", "Failed to write to #{File.join(site.dest, filename)}: #{e.message}"
        end
        
        # Tạo thêm file trong source để có thể access từ Jekyll
        begin
          File.open(File.join(site.source, filename), 'w') do |file|
            file.write(JSON.pretty_generate(search_data))
          end
        rescue => e
          Jekyll.logger.warn "Search Generator", "Failed to write to #{File.join(site.source, filename)}: #{e.message}"
        end
      end
    end
  end
end
