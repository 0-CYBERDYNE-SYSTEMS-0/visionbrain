
import re
p='site/index-v2.html'
s=open(p).read()
for tag in ['html','head','body','style','script','main','section','svg','defs','g','table','dl','figure','tr','ul','ol']:
    o=len(re.findall(r'<%s[ >]'%tag,s)); c=len(re.findall(r'</%s>'%tag,s))
    if o!=c:
        print(f'  MISMATCH {tag}: open={o} close={c}')
print('tag balance: checked')
print('bytes:', len(s))
print('viewBoxes:', re.findall(r'viewBox="([^"]+)"', s))
print('plates:', len(re.findall(r'<svg role="img"', s)))
print('h2:', re.findall(r'<h2>([^<]+)</h2>', s))
print('transition:all?', 'transition: all' in s)
print('scale(0)?', 'scale(0)' in s)
print("scroll listener?", "addEventListener('scroll'" in s)
