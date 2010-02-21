#!/bin/bash
#
# Generate a XML sitemap to make the search engines love the website.
#

: ${baseurl:="http://www.pymvpa.org"}
siteroot="build/website"

cat << EOT
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url>
<loc>$baseurl/</loc>
<lastmod>$(stat -c '%y' $siteroot/index.html | cut -d ' ' -f 1,1)</lastmod>
<priority>1.0</priority>
</url>
EOT

for file in $(find $siteroot/ -maxdepth 1 -name '*.html' -o -name 'PyMVPA*.pdf'); do
cat << EOT
<url>
<loc>$baseurl/$(basename $file)</loc>
<lastmod>$(stat -c '%y' $file | cut -d ' ' -f 1,1)</lastmod>
</url>
EOT
done
echo "</urlset>"


