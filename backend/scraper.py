import yt_dlp
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
from config import Config
import logging

logger = logging.getLogger(__name__)

class SocialMediaScraper:
    def __init__(self):
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'socket_timeout': Config.REQUEST_TIMEOUT,
        }
    
    def extract_media(self, url):
        """Extract media URL from social media post"""
        platform = self._detect_platform(url)
        logger.info(f"Detected platform: {platform}")
        
        try:
            if platform in ['instagram', 'facebook']:
                return self._extract_meta_media(url, platform)
            elif platform in ['twitter', 'x']:
                return self._extract_twitter_media(url)
            elif platform in ['tiktok']:
                return self._extract_tiktok_media(url)
            else:
                return self._extract_generic_media(url)
        except Exception as e:
            logger.error(f"Failed to extract media from {platform}: {str(e)}")
            raise
    
    def _detect_platform(self, url):
        """Detect social media platform from URL"""
        domain = urlparse(url).netloc.lower()
        
        platform_map = {
            'instagram.com': 'instagram',
            'facebook.com': 'facebook',
            'fb.com': 'facebook',
            'twitter.com': 'twitter',
            'x.com': 'twitter',
            'tiktok.com': 'tiktok',
            'youtube.com': 'youtube',
            'youtu.be': 'youtube',
        }
        
        for key, value in platform_map.items():
            if key in domain:
                return value
        return 'generic'
    
    def _extract_meta_media(self, url, platform):
        """Extract media from Instagram/Facebook"""
        headers = {'User-Agent': Config.USER_AGENT}
    
        # Try yt-dlp first for video, but catch errors
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'ignoreerrors': True,
                'socket_timeout': Config.REQUEST_TIMEOUT,
                'format': 'best',
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            
                if info:
                    if 'entries' in info and info['entries']:
                        info = info['entries'][0]
                
                    if info and info.get('url') and info.get('vcodec') and info['vcodec'] != 'none':
                        return {
                            'media_urls': [info['url']],
                            'media_type': 'video',
                            'platform': platform,
                            'title': info.get('title', ''),
                            'description': info.get('description', '')
                        }
        except:
            pass
    
        # Fallback: Extract og:image from the page (works for Instagram images)
        try:
            response = requests.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT)
            soup = BeautifulSoup(response.content, 'html.parser')
        
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                return {
                    'media_urls': [og_image['content']],
                    'media_type': 'image',
                    'platform': platform,
                    'title': '',
                    'description': ''
                }
        except:
            pass
    
        raise ValueError("No media found. Instagram may require login. Try a public post URL.")
    
    def _extract_twitter_media(self, url):
        """Extract media from Twitter/X"""
        headers = {'User-Agent': Config.USER_AGENT}
        
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if 'entries' in info:
                    info = info['entries'][0]
                
                media_urls = []
                if info.get('url'):
                    media_urls.append(info['url'])
                elif info.get('thumbnails'):
                    thumbnails = sorted(
                        info['thumbnails'], 
                        key=lambda x: x.get('height', 0) or 0, 
                        reverse=True
                    )
                    if thumbnails:
                        media_urls.append(thumbnails[0]['url'])
                
                if media_urls:
                    return {
                        'media_urls': media_urls,
                        'media_type': 'image',
                        'platform': 'twitter',
                        'title': info.get('title', ''),
                        'description': info.get('description', '')
                    }
        except:
            pass
        
        response = requests.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            return {
                'media_urls': [og_image['content']],
                'media_type': 'image',
                'platform': 'twitter',
                'title': '',
                'description': ''
            }
        
        raise ValueError("Could not extract media from Twitter post")
    
    def _extract_tiktok_media(self, url):
        """Extract media from TikTok"""
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            if 'entries' in info:
                info = info['entries'][0]
            
            media_urls = []
            if info.get('url'):
                media_urls.append(info['url'])
            
            if not media_urls:
                raise ValueError("No video found in TikTok post")
            
            return {
                'media_urls': media_urls,
                'media_type': 'video',
                'platform': 'tiktok',
                'title': info.get('title', ''),
                'description': info.get('description', '')
            }
    
    def _extract_generic_media(self, url):
        """Generic media extraction for other platforms"""
        headers = {'User-Agent': Config.USER_AGENT}
        response = requests.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        og_media = soup.find('meta', property='og:video') or \
                  soup.find('meta', property='og:image')
        
        if og_media and og_media.get('content'):
            media_type = 'video' if 'og:video' in str(og_media) else 'image'
            return {
                'media_urls': [og_media['content']],
                'media_type': media_type,
                'platform': 'generic',
                'title': '',
                'description': ''
            }
        
        raise ValueError("Could not extract media from URL")
    
    def download_media(self, media_url, save_path=None):
        """Download media file from URL"""
        headers = {'User-Agent': Config.USER_AGENT}
        response = requests.get(media_url, headers=headers, 
                               timeout=Config.REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > Config.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {int(content_length)} bytes")
        
        if save_path:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return save_path
        
        return response.content