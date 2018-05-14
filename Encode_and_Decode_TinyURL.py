import base64


class Codec:
    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.

        :type longUrl: str
        :rtype: str
        """
        return longUrl.__hash__()


    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.

        :type shortUrl: str
        :rtype: str
        """
        out = base64.decodestring(shortUrl)

        print(out)


url = 'https://leetcode.com/problems/design-tinyurl'
codec = Codec()
codec.decode(codec.encode(url))
